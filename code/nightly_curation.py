"""Source-grounded curation helpers for nightly memory consolidation."""
from __future__ import annotations
import hashlib, json, re
from datetime import datetime
from memory import _is_valid_person_tag


class NightlyCurator:
    def __init__(self, memory):
        self.memory = memory

    @staticmethod
    def _mid(moment):
        return str((moment or {}).get("id") or "")

    @staticmethod
    def _video(moment):
        return str(((moment or {}).get("source_provenance") or {}).get("source_video_id") or "")

    @staticmethod
    def _rows(moment):
        rows = (moment or {}).get("person_observations") or []
        return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []

    @classmethod
    def _names(cls, moment):
        raw = (moment or {}).get("people") or []
        if isinstance(raw, str):
            raw = [raw]
        return ({str(value).strip().casefold() for value in raw if isinstance(value, str) and value.strip()}
                | {str(row.get("entity") or "").strip().casefold()
                   for row in cls._rows(moment) if row.get("entity")})

    @classmethod
    def _spoken_aliases(cls, moment):
        result = set()
        for row in cls._rows(moment):
            for alias in row.get("spoken_aliases") or []:
                if isinstance(alias, dict) and alias.get("status") == "probable_visual_match":
                    text = str(alias.get("alias") or "").strip().casefold()
                    if text:
                        result.add(text)
        return result

    @classmethod
    def _description(cls, moment, entity=None):
        rows = cls._rows(moment)
        if entity:
            rows = [row for row in rows if str(row.get("entity") or "").strip().casefold() == entity.casefold()]
        return " ".join(str(row.get(key) or "") for row in rows
                        for key in ("appearance", "face_features", "profile_context", "relationships"))

    @staticmethod
    def _tokens(value):
        stop = {"the","and","with","wearing","woman","person","visible","standing","smiling",
                "appears","wears","this","that","from","into","她","这个","一个","人物","画面","视频","场景"}
        tokens = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", str(value or "").casefold())) - stop
        output = set()
        for token in tokens:
            if re.fullmatch(r"[\u4e00-\u9fff]+", token):
                output.update(token[i:i+2] for i in range(len(token)-1))
            else:
                output.add(token.replace("grey", "gray"))
        return output

    @classmethod
    def _overlap(cls, a, b):
        return len(cls._tokens(a) & cls._tokens(b))

    @classmethod
    def _scene_overlap(cls, left, right):
        fields = ("scene", "location", "activity", "environments")
        return cls._overlap(" ".join(str(left.get(k) or "") for k in fields),
                            " ".join(str(right.get(k) or "") for k in fields))

    @staticmethod
    def _family(moment):
        video = NightlyCurator._video(moment)
        return video.rsplit("/", 1)[0] if "/" in video else ""

    @staticmethod
    def _is_clip_local_person(value):
        return bool(re.fullmatch(
            r"(?:woman|man|person|people|人物|人物编号)\s*#?\s*\d+",
            str(value or "").strip(), re.I))

    @classmethod
    def _speech_anchor_rows(cls, moment):
        provenance = (moment or {}).get("source_provenance") or {}
        segment = str(provenance.get("source_segment_id") or "")
        if not segment:
            return []
        anchors = []
        for row in cls._rows(moment):
            entity = str(row.get("entity") or "").strip()
            if not entity:
                continue
            for alias in row.get("spoken_aliases") or []:
                if not isinstance(alias, dict) or alias.get("status") != "probable_visual_match":
                    continue
                canonical = str(alias.get("alias") or "").strip()
                speech_ref = alias.get("evidence_ref") or {}
                visual_ref = alias.get("visual_evidence_ref") or {}
                if (not canonical or not _is_valid_person_tag(canonical)
                        or speech_ref.get("modality") != "speech"
                        or visual_ref.get("modality") != "visual"
                        or str(speech_ref.get("source_segment_id") or "") != segment
                        or str(visual_ref.get("source_segment_id") or "") != segment):
                    continue
                anchors.append((entity, canonical, row))
        return anchors

    @classmethod
    def _automatic_identity_links(cls, moments):
        """Build auditable identity clusters from repeated speech-grounded anchors."""
        by_family_alias = {}
        for moment in moments:
            family = cls._family(moment)
            if not family:
                continue
            by_moment = {}
            for entity, canonical, row in cls._speech_anchor_rows(moment):
                by_moment.setdefault(canonical.casefold(), []).append((entity, canonical, row))
            for alias_key, rows in by_moment.items():
                if len({row[0].casefold() for row in rows}) != 1:
                    continue
                entity, canonical, observation = rows[0]
                by_family_alias.setdefault((family, alias_key), {
                    "canonical": canonical, "anchors": []
                })["anchors"].append((moment, entity, observation))

        links = []
        moments_by_family = {}
        for moment in moments:
            family = cls._family(moment)
            if family:
                moments_by_family.setdefault(family, []).append(moment)

        candidate_options = {}
        for (family, _alias_key), cluster in sorted(by_family_alias.items()):
            anchors = cluster["anchors"]
            videos = {}
            for moment, entity, observation in anchors:
                video = cls._video(moment)
                if video:
                    videos.setdefault(video, []).append((moment, entity, observation))
            if len(videos) < 2:
                continue
            canonical = cluster["canonical"]
            anchor_ids = sorted({cls._mid(moment) for moment, _, _ in anchors if cls._mid(moment)})
            if len(anchor_ids) < 2:
                continue

            for moment, entity, _observation in anchors:
                evidence_ids = sorted({cls._mid(other) for other, other_entity, _ in anchors
                                       if cls._mid(other) and cls._mid(other) != cls._mid(moment)
                                       and cls._video(other) != cls._video(moment)
                                       and other_entity.casefold() == canonical.casefold()})
                if evidence_ids:
                    links.append({
                        "moment_id": cls._mid(moment), "observed_entity": entity,
                        "canonical_entity": canonical, "confidence": 0.98,
                        "evidence_moment_ids": evidence_ids[:6],
                        "reason": "Repeated speech-to-visual identity anchors in separate source clips.",
                    })

            candidates = candidate_options
            for moment in moments_by_family.get(family, []):
                rows = cls._rows(moment)
                if len(rows) != 1:
                    continue
                observation = rows[0]
                old = str(observation.get("entity") or "").strip()
                if (not cls._is_clip_local_person(old)
                        or str(observation.get("confidence") or "").casefold() != "high"
                        or (cls._mid(moment), old.casefold()) in {
                            (cls._mid(anchor_moment), entity.casefold())
                            for anchor_moment, entity, _ in anchors
                        }):
                    continue
                matches = []
                for video in sorted(videos):
                    best = None
                    for anchor_moment, _entity, anchor_observation in videos[video]:
                        appearance = cls._overlap(
                            observation.get("appearance"), anchor_observation.get("appearance"))
                        face = cls._overlap(
                            observation.get("face_features"), anchor_observation.get("face_features"))
                        scene = cls._scene_overlap(moment, anchor_moment)
                        if appearance < 3 or scene < 3:
                            continue
                        score = appearance + face * 2 + min(scene, 8) * 0.25
                        if best is None or score > best[0]:
                            best = (score, face, cls._mid(anchor_moment))
                    if best:
                        matches.append(best)
                if len(matches) < 2 or max(match[1] for match in matches) < 2:
                    continue
                evidence_ids = sorted({match[2] for match in matches if match[2]})
                if len(evidence_ids) < 2:
                    continue
                score = sum(match[0] for match in matches)
                candidates.setdefault(cls._mid(moment), []).append({
                    "score": score, "old": old, "canonical": canonical,
                    "evidence_moment_ids": evidence_ids[:6],
                })

        for mid, options in candidate_options.items():
            options.sort(key=lambda item: (-item["score"], item["canonical"].casefold()))
            if len(options) > 1 and options[0]["score"] - options[1]["score"] < 3:
                continue
            selected = options[0]
            links.append({
                "moment_id": mid, "observed_entity": selected["old"],
                "canonical_entity": selected["canonical"],
                "confidence": min(0.96, 0.88 + max(0, selected["score"] - 10) * 0.005),
                "evidence_moment_ids": selected["evidence_moment_ids"],
                "reason": "One visible person matches two independent speech-grounded identity anchors by face features, appearance, and scene.",
            })
        return links

    @classmethod
    def _image_description(cls, moment, entity):
        name = str(entity or "").strip().casefold()
        descriptions = []
        for row in cls._rows(moment):
            if name and str(row.get("entity") or "").strip().casefold() == name:
                value = " ".join(str(row.get(key) or "") for key in
                                 ("appearance", "face_features", "profile_context"))
                if value.strip():
                    descriptions.append(value.strip())
        media = (moment or {}).get("multimodal") or {}
        for group_key in ("person_visuals", "object_visuals", "entity_visuals"):
            group = media.get(group_key) or {}
            for item in group.get("items") or []:
                if not isinstance(item, dict):
                    continue
                label = str(item.get("person") or item.get("object") or
                            item.get("entity") or "").strip().casefold()
                if name and label == name:
                    value = str(item.get("description") or item.get("appearance") or "")
                    if value.strip():
                        descriptions.append(value.strip())
        return " ".join(dict.fromkeys(descriptions))[:500]

    @classmethod
    def _image_refs(cls, moment):
        mid, media, result = cls._mid(moment), (moment or {}).get("multimodal") or {}, []
        face_labels = {}
        for item in ((media.get("person_visuals") or {}).get("items") or []):
            if not isinstance(item, dict):
                continue
            label = str(item.get("person") or "").strip().casefold()
            for ref in item.get("frame_refs") or []:
                if (label and isinstance(ref, dict) and ref.get("visual_role") == "face_crop"
                        and ref.get("identity_verified") is True and ref.get("asset")):
                    face_labels.setdefault(str(ref["asset"]), set()).add(label)
        ambiguous_faces = {asset for asset, labels in face_labels.items() if len(labels) > 1}
        def add(ref, role="", entity=""):
            if not isinstance(ref, dict) or not ref.get("asset"):
                return
            role = str(ref.get("visual_role") or role)
            if role == "face_crop" and (
                ref.get("identity_verified") is not True
                or not str(ref.get("identity_assignment") or "").startswith("qwen_visual_anchor+")
                or str(ref.get("asset") or "") in ambiguous_faces
            ):
                return
            if role not in {"face_crop","person_crop","object_crop","qwen_input_frame","location_source_frame","scene_crop","context_crop"}:
                return
            if role == "object_crop" and not entity:
                return
            result.append((f"{mid}#image:{len(result)}", ref, role, str(entity or "")))
        for ref in media.get("visual_frames") or []:
            add(ref, "qwen_input_frame")
        if media.get("visual"):
            add(media["visual"], "qwen_input_frame")
        for group_key in ("person_visuals","object_visuals","entity_visuals"):
            group = media.get(group_key) or {}
            for item in group.get("items") or []:
                if not isinstance(item, dict):
                    continue
                entity = item.get("person") or item.get("object") or item.get("entity") or ""
                for ref in item.get("frame_refs") or []:
                    add(ref, "", entity)
            if group_key == "entity_visuals":
                for ref in group.get("assets") or []:
                    add(ref)
        return result

    @classmethod
    def evidence_bundle(cls, moments):
        rows, images = [], []
        for moment in moments:
            mid = cls._mid(moment)
            if not mid:
                continue
            provenance = moment.get("source_provenance") or {}
            observations = []
            for person in cls._rows(moment)[:8]:
                aliases = []
                for alias in person.get("spoken_aliases") or []:
                    if isinstance(alias, dict):
                        aliases.append({k: alias.get(k) for k in
                            ("alias","status","confidence","basis","evidence_ref","visual_evidence_ref")
                            if alias.get(k) not in (None,"")})
                observations.append({k: person.get(k) for k in
                    ("entity","appearance","face_features","profile_context","relationships","confidence")
                    if person.get(k) not in (None,"")})
                observations[-1]["spoken_aliases"] = aliases[:4]
            speech = []
            for fact in moment.get("temporal_facts") or []:
                if isinstance(fact, dict) and (
                    str(fact.get("modality") or "").casefold() == "speech" or
                    str(fact.get("subject_type") or "").casefold() == "person"
                ):
                    speech.append({k: fact.get(k) for k in (
                        "subject_type","subject","actor","predicate","value","evidence",
                        "confidence","modality","evidence_ref","evidence_refs")
                        if fact.get(k) not in (None,"")})
            row = {
                "moment_id": mid, "source_video_id": provenance.get("source_video_id",""),
                "source_segment_id": provenance.get("source_segment_id",""),
                "source_time": moment.get("source_time") or {},
                "scene": str(moment.get("scene") or moment.get("user_action") or "")[:500],
                "interaction_details": str(moment.get("interaction_details") or "")[:500],
                "location": moment.get("location") or "", "activity": moment.get("activity") or "",
                "people": moment.get("people") or [], "person_observations": observations,
                "speech_and_person_facts": speech[:12],
                "source_evidence": str(moment.get("source_evidence") or "")[:1000],
                "objects": moment.get("objects") or [],
                "object_states": (moment.get("object_states") or [])[:8],
            }
            ref_ids = []
            for ref_id, ref, role, entity in cls._image_refs(moment):
                images.append({"ref_id": ref_id, "visual_role": role, "entity": entity,
                               "description": cls._image_description(moment, entity),
                               "at_s": ref.get("at_s", ref.get("frame_time_s")),
                               "identity_verified": ref.get("identity_verified") is True,
                               "quality_tier": ref.get("quality_tier", "source")})
                ref_ids.append(ref_id)
            row["image_ref_ids"] = ref_ids[:16]
            rows.append(row)
        if len(rows) > 100:
            ranked = sorted(rows, key=lambda r: (
                -bool(r["person_observations"]),
                -bool(r["speech_and_person_facts"] or r["source_evidence"]), r["moment_id"]))
            keep = {r["moment_id"] for r in ranked[:100]}
            rows = [r for r in rows if r["moment_id"] in keep]
            images = [r for r in images if r["ref_id"].split("#image:",1)[0] in keep]
        return {"moments": rows, "image_candidates": images[:1200]}

    def validate(self, rollup, moments):
        archive = {self._mid(m): m for m in moments if self._mid(m)}
        ids, links, claimed = set(archive), [], set()
        candidates = self._automatic_identity_links(moments)
        candidates.extend(rollup.get("identity_resolutions") or [])
        for item in candidates:
            if not isinstance(item, dict):
                continue
            mid, old, new = (str(item.get(k) or "").strip() for k in
                             ("moment_id","observed_entity","canonical_entity"))
            target = archive.get(mid)
            if not target or not old or not new:
                continue
            if not _is_valid_person_tag(new) or old.casefold() not in self._names(target):
                continue
            try:
                score = max(0.0, min(1.0, float(item.get("confidence") or 0)))
            except (TypeError, ValueError):
                continue
            if score < .82:
                continue
            cited = list(dict.fromkeys(str(x) for x in item.get("evidence_moment_ids") or []
                                       if str(x) in ids and str(x) != mid))
            anchors = [archive[x] for x in cited if
                       new.casefold() in self._names(archive[x]) or
                       new.casefold() in self._spoken_aliases(archive[x])]
            if not anchors:
                continue
            direct = new.casefold() in self._spoken_aliases(target)
            if not direct:
                if len(anchors) < 2:
                    continue
                group = self._video(target).rsplit("/",1)[0]
                anchors = [a for a in anchors if group and self._video(a).rsplit("/",1)[0] == group]
                if len(anchors) < 2:
                    continue
                if self._overlap(self._description(target, old),
                                 " ".join(self._description(a, new) for a in anchors)) < 2:
                    continue
                if max(self._scene_overlap(target, a) for a in anchors) < 2:
                    continue
            key = (mid, old.casefold())
            if key in claimed:
                continue
            previous = target.get("nightly_identity_resolutions") or []
            if any(isinstance(x,dict) and str(x.get("observed_entity") or "").casefold()==old.casefold()
                   and str(x.get("canonical_entity") or "").casefold()!=new.casefold() for x in previous):
                continue
            claimed.add(key)
            links.append({"moment_id":mid,"observed_entity":old,"canonical_entity":new,
                          "confidence":score,"reason":str(item.get("reason") or "")[:360],
                          "evidence_moment_ids":cited})

        highlights = []
        for item in rollup.get("highlight_moments") or []:
            if not isinstance(item,dict):
                continue
            mid, reason = str(item.get("moment_id") or ""), str(item.get("reason") or "").strip()[:360]
            try: score = max(0.0,min(1.0,float(item.get("score") or 0)))
            except (TypeError,ValueError): continue
            if mid in ids and score >= .82 and reason:
                highlights.append({"moment_id":mid,"score":score,"reason":reason,
                    "evidence_moment_ids":[str(x) for x in item.get("evidence_moment_ids") or [] if str(x) in ids]})
        highlights = sorted(highlights,key=lambda x:(-x["score"],x["moment_id"]))[:8]
        image_map = {ref_id:ref for m in moments for ref_id,ref,_,_ in self._image_refs(m)}
        image_marks = []
        for item in rollup.get("highlight_images") or []:
            if not isinstance(item,dict):
                continue
            ref_id, reason = str(item.get("ref_id") or ""), str(item.get("reason") or "").strip()[:360]
            try: score = max(0.0,min(1.0,float(item.get("score") or 0)))
            except (TypeError,ValueError): continue
            if ref_id in image_map and score >= .82 and reason:
                image_marks.append({"ref_id":ref_id,"score":score,"reason":reason,
                                    "moment_id":ref_id.split("#image:",1)[0]})
        image_marks = sorted(image_marks,key=lambda x:(-x["score"],x["ref_id"]))[:12]
        return {"identity_resolutions":links,"highlight_moments":highlights,
                "highlight_images":image_marks,"image_ref_map":image_map}

    def apply_identity_resolutions(self, links):
        rewrites, now = [], datetime.now().astimezone().isoformat(timespec="seconds")
        archive = self.memory.memory.get("layer7",{}).get("moments",{})
        for link in links:
            mid, old, new = link["moment_id"], link["observed_entity"], link["canonical_entity"]
            moment = archive.get(mid)
            if not isinstance(moment,dict):
                continue
            existing_links = moment.setdefault("nightly_identity_resolutions", [])
            evidence_ids = sorted(set(link.get("evidence_moment_ids") or []))
            if any(isinstance(existing, dict)
                   and str(existing.get("observed_entity") or "").casefold() == old.casefold()
                   and str(existing.get("canonical_entity") or "").casefold() == new.casefold()
                   and sorted(set(existing.get("evidence_moment_ids") or [])) == evidence_ids
                   for existing in existing_links):
                continue
            matched = []
            for row in self._rows(moment):
                if str(row.get("entity") or "").strip().casefold() != old.casefold():
                    continue
                row.setdefault("source_entity",old)
                row["entity"] = new
                audit = {"source":"nightly_evidence_reconciliation","observed_entity":old,
                         "canonical_entity":new,"confidence":link["confidence"],
                         "reason":link["reason"],"evidence_moment_ids":link["evidence_moment_ids"],
                         "resolved_at":now}
                row["identity_resolution"] = audit
                row.setdefault("identity_resolution_history",[]).append(audit)
                matched.append(row)
            raw = moment.get("people") or []
            if isinstance(raw,str): raw=[raw]
            people=[]
            for value in raw:
                value = new if isinstance(value,str) and value.strip().casefold()==old.casefold() else value
                if value not in people: people.append(value)
            remaining={str(row.get("entity") or "").casefold() for row in self._rows(moment) if row not in matched}
            if old.casefold() not in remaining:
                people=[x for x in people if not (isinstance(x,str) and x.strip().casefold()==old.casefold())]
            if new not in people: people.append(new)
            moment["people"] = people
            for fact in moment.get("temporal_facts") or []:
                if not isinstance(fact, dict):
                    continue
                if (str(fact.get("subject_type") or "").casefold() == "person"
                        and str(fact.get("subject") or "").strip().casefold() == old.casefold()):
                    fact["subject"] = new
                if str(fact.get("actor") or "").strip().casefold() == old.casefold():
                    fact["actor"] = new
                for field in ("affected_people", "people"):
                    values = fact.get(field)
                    if isinstance(values, list):
                        fact[field] = list(dict.fromkeys(
                            new if isinstance(value, str) and value.strip().casefold() == old.casefold()
                            else value for value in values))
            media=moment.get("multimodal") or {}
            for group_key in ("person_visuals","entity_visuals"):
                group=media.get(group_key) or {}
                for visual in group.get("items") or []:
                    if not isinstance(visual,dict): continue
                    if group_key == "entity_visuals" and visual.get("entity_type") != "person":
                        continue
                    label_key="person" if group_key=="person_visuals" else "entity"
                    current=str(visual.get(label_key) or visual.get("entity") or "").strip()
                    if current.casefold()==old.casefold():
                        visual.setdefault("source_entity",old)
                        visual[label_key]=new
                        visual["identity_resolution"]={"source":"nightly_evidence_reconciliation",
                            "confidence":link["confidence"],"evidence_moment_ids":link["evidence_moment_ids"]}
            existing_links.append({**link,"resolved_at":now})
            self.memory._sync_window_moment(mid)
            rewrites.append((mid,old,new))
        return rewrites

    def remove_old_index_memberships(self, rewrites):
        index=self.memory.memory.get("layer7",{}).get("people",{})
        for mid,old,_new in rewrites:
            ids=index.get(old)
            if isinstance(ids,list):
                ids[:]=[value for value in ids if value!=mid]
                if not ids: index.pop(old,None)

    @staticmethod
    def exact_duplicate_pairs(moments):
        """Merge only exact repeat writes of the same source segment."""
        buckets={}
        for moment in moments:
            if not isinstance(moment,dict) or moment.get("merged_into") or moment.get("highlighted") or moment.get("nightly_importance"):
                continue
            provenance=moment.get("source_provenance") or {}
            video=str(provenance.get("source_video_id") or "")
            segment=str(provenance.get("source_segment_id") or "")
            if not video or not segment: continue
            facts={key:moment.get(key) for key in (
                "scene","user_action","interaction_details","source_evidence","people",
                "person_observations","location","activity","objects","object_states","temporal_facts")}
            if not any(facts.get(k) for k in ("scene","user_action","interaction_details","source_evidence","temporal_facts")):
                continue
            sig=hashlib.sha256(json.dumps(facts,ensure_ascii=False,sort_keys=True,
                separators=(",",":")).encode()).hexdigest()
            buckets.setdefault((video,segment,sig),[]).append(moment)
        pairs=[]
        for rows in buckets.values():
            if len(rows)<2: continue
            rows.sort(key=lambda x:(str(x.get("timestamp") or ""),str(x.get("id") or "")))
            target=NightlyCurator._mid(rows[0])
            pairs.extend((NightlyCurator._mid(source),target) for source in rows[1:])
        return pairs

    def apply_importance(self, highlights, image_marks, image_map, scope_moment_ids=None):
        now=datetime.now().astimezone().isoformat(timespec="seconds")
        archive=self.memory.memory.get("layer7",{}).get("moments",{})
        changed=0
        # Nightly highlights are a rolling editorial selection, not permanent
        # stickers. Recompute only the moments included in this consolidation;
        # manual/user highlights use a different source and remain untouched.
        scope = {str(mid) for mid in (scope_moment_ids or []) if str(mid) in archive}
        for mid in scope:
            moment = archive.get(mid)
            if not isinstance(moment, dict):
                continue
            removed = False
            if (moment.get("nightly_importance") or {}).get("source") == "nightly_consolidation":
                moment.pop("nightly_importance", None)
                removed = True
            for _ref_id, ref, _role, _entity in self._image_refs(moment):
                if (ref.get("nightly_importance") or {}).get("source") == "nightly_consolidation":
                    ref.pop("nightly_importance", None)
                    removed = True
            if removed:
                self.memory._sync_window_moment(mid)
                changed += 1
        for item in highlights:
            moment=archive.get(item["moment_id"])
            if not isinstance(moment,dict): continue
            mark={"score":item["score"],"reason":item["reason"],
                  "source_moment_ids":item["evidence_moment_ids"],
                  "source":"nightly_consolidation","marked_at":now}
            old_mark = moment.get("nightly_importance") or {}
            same_mark = (old_mark.get("score") == mark["score"]
                         and old_mark.get("reason") == mark["reason"]
                         and old_mark.get("source") == mark["source"])
            if not same_mark:
                moment["nightly_importance"]=mark
                self.memory._sync_window_moment(item["moment_id"]); changed+=1
        for item in image_marks:
            ref=image_map.get(item["ref_id"]); moment=archive.get(item["moment_id"])
            if not isinstance(ref,dict) or not isinstance(moment,dict): continue
            mark={"score":item["score"],"reason":item["reason"],
                  "source":"nightly_consolidation","marked_at":now}
            old_mark = ref.get("nightly_importance") or {}
            same_mark = (old_mark.get("score") == mark["score"]
                         and old_mark.get("reason") == mark["reason"]
                         and old_mark.get("source") == mark["source"])
            if not same_mark:
                ref["nightly_importance"]=mark
                self.memory._sync_window_moment(item["moment_id"])
                changed+=1
        if changed: self.memory._save()
        return changed
