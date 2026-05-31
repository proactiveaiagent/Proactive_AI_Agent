import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


class PersonMemory:
    """Hierarchical memory system with automatic consolidation"""

    def __init__(self, memory_dir="memory"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)

        self.memory_file = self.memory_dir / "memory.json"
        self.memory: Dict = self._load_memory()

    def _load_memory(self) -> Dict:
        """Load memory from disk"""
        if self.memory_file.exists():
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            "people": {},  # {person_name: {locations: [], last_seen: "", notes: "", layer: "episodic"}}
            "locations": {},
            # {location: {people: [], last_visit: "", notes: "", layer: "episodic", canonical_name: ""}}
            "relationships": {},  # {person: {relation_type: [other_people]}}
            "semantic_facts": [],  # [{fact: "", confidence: 0-1, source_count: int}]
            "metadata": {
                "total_encounters": 0,
                "last_consolidation": None
            }
        }

    def _save_memory(self):
        """Save memory to disk"""
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(self.memory, f, indent=2, ensure_ascii=False)

    def update(self, people: List[str], location: str, notes: Optional[str] = None):
        """Update memory with people and location"""
        timestamp = datetime.now().isoformat()

        # Update people records
        for person in people:
            if person not in self.memory["people"]:
                self.memory["people"][person] = {
                    "locations": [],
                    "last_seen": timestamp,
                    "notes": "",
                    "layer": "episodic",  # episodic -> semantic -> core
                    "encounter_count": 0
                }

            # Add location if not already there
            if location not in self.memory["people"][person]["locations"]:
                self.memory["people"][person]["locations"].append(location)

            self.memory["people"][person]["last_seen"] = timestamp
            self.memory["people"][person]["encounter_count"] += 1

            if notes:
                self.memory["people"][person]["notes"] = notes

        # Update location records
        if location not in self.memory["locations"]:
            self.memory["locations"][location] = {
                "people": [],
                "last_visit": timestamp,
                "notes": "",
                "layer": "episodic",
                "canonical_name": location,
                "visit_count": 0
            }

        for person in people:
            if person not in self.memory["locations"][location]["people"]:
                self.memory["locations"][location]["people"].append(person)

        self.memory["locations"][location]["last_visit"] = timestamp
        self.memory["locations"][location]["visit_count"] += 1
        self.memory["metadata"]["total_encounters"] += 1

        self._save_memory()

    def get_context(self, people: List[str] = None, location: str = None) -> str:
        """Generate memory context for the LLM"""
        context_parts = []

        if people:
            for person in people:
                if person in self.memory["people"]:
                    person_data = self.memory["people"][person]
                    locations = ", ".join(person_data["locations"])
                    context_parts.append(
                        f"- {person} [{person_data['layer']}]: Previously seen at {locations}. "
                        f"Last seen: {person_data['last_seen'][:10]}. "
                        f"Encounters: {person_data['encounter_count']}. "
                        f"{person_data['notes']}"
                    )

        if location and location in self.memory["locations"]:
            loc_data = self.memory["locations"][location]
            people_here = ", ".join(loc_data["people"])
            context_parts.append(
                f"- Location '{loc_data['canonical_name']}' [{loc_data['layer']}]: "
                f"Previously met {people_here} here. "
                f"Last visit: {loc_data['last_visit'][:10]}. "
                f"Visits: {loc_data['visit_count']}. "
                f"{loc_data['notes']}"
            )

        if not context_parts:
            return "No previous memory found."

        return "Memory Context:\n" + "\n".join(context_parts)

    def get_all_memory(self) -> str:
        """Get formatted summary of all memory with hierarchy"""
        summary = ["=== MEMORY SUMMARY ===\n"]

        # Group people by layer
        layers = {"core": [], "semantic": [], "episodic": []}
        for person, data in self.memory["people"].items():
            layer = data.get("layer", "episodic")
            layers[layer].append((person, data))

        summary.append(f"Known People: {len(self.memory['people'])}")
        for layer in ["core", "semantic", "episodic"]:
            if layers[layer]:
                summary.append(f"\n  [{layer.upper()}]")
                for person, data in layers[layer]:
                    locations_str = ', '.join(data['locations'][:3])
                    if len(data['locations']) > 3:
                        locations_str += f" (+{len(data['locations']) - 3} more)"
                    summary.append(f"  • {person}: {locations_str}")

        # Group locations by layer
        loc_layers = {"core": [], "semantic": [], "episodic": []}
        for location, data in self.memory["locations"].items():
            layer = data.get("layer", "episodic")
            loc_layers[layer].append((location, data))

        summary.append(f"\nKnown Locations: {len(self.memory['locations'])}")
        for layer in ["core", "semantic", "episodic"]:
            if loc_layers[layer]:
                summary.append(f"\n  [{layer.upper()}]")
                for location, data in loc_layers[layer]:
                    canonical = data.get("canonical_name", location)
                    people_str = ', '.join(data['people'])
                    summary.append(f"  • {canonical}: {people_str}")

        # Add relationships if any
        if self.memory.get("relationships"):
            summary.append(f"\nRelationships:")
            for person, relations in self.memory["relationships"].items():
                for rel_type, related_people in relations.items():
                    summary.append(f"  • {person} ({rel_type}): {', '.join(related_people)}")

        # Add semantic facts if any
        if self.memory.get("semantic_facts"):
            summary.append(f"\nSemantic Facts: {len(self.memory['semantic_facts'])}")
            for fact in self.memory["semantic_facts"][:5]:
                summary.append(f"  • {fact['fact']} (confidence: {fact['confidence']:.2f})")

        return "\n".join(summary)

    def consolidate(self, consolidation_analysis: dict):
        """Apply consolidation results from LLM analysis"""

        # Update people
        if "people" in consolidation_analysis:
            for person_name, updates in consolidation_analysis["people"].items():
                if person_name in self.memory["people"]:
                    # Preserve existing data while updating
                    for key, value in updates.items():
                        if key == "canonical_locations":
                            # Replace locations list with canonical versions
                            self.memory["people"][person_name]["locations"] = value
                        elif key == "relationship":
                            # Store relationship info in notes or separate field
                            if "relationship" not in self.memory["people"][person_name]:
                                self.memory["people"][person_name]["relationship"] = value
                        else:
                            self.memory["people"][person_name][key] = value

        # Update locations with deduplication
        if "locations" in consolidation_analysis:
            locations_to_delete = []

            for old_loc, new_data in consolidation_analysis["locations"].items():
                canonical = new_data.get("canonical_name", old_loc)

                # If this is a merge operation
                if canonical != old_loc:
                    # Create or update canonical location
                    if canonical not in self.memory["locations"]:
                        # Start with old location data if it exists
                        if old_loc in self.memory["locations"]:
                            self.memory["locations"][canonical] = self.memory["locations"][old_loc].copy()
                        else:
                            self.memory["locations"][canonical] = {
                                "people": [],
                                "last_visit": datetime.now().isoformat(),
                                "notes": "",
                                "layer": "episodic",
                                "canonical_name": canonical,
                                "visit_count": 0
                            }

                    # Merge people from old location to canonical
                    if old_loc in self.memory["locations"]:
                        old_people = set(self.memory["locations"][old_loc].get("people", []))
                        new_people = set(self.memory["locations"][canonical].get("people", []))
                        self.memory["locations"][canonical]["people"] = list(old_people | new_people)

                        # Merge visit counts
                        self.memory["locations"][canonical]["visit_count"] = (
                                self.memory["locations"][canonical].get("visit_count", 0) +
                                self.memory["locations"][old_loc].get("visit_count", 0)
                        )

                        # Mark old location for deletion
                        if old_loc != canonical:
                            locations_to_delete.append(old_loc)

                    # Update with new data
                    self.memory["locations"][canonical].update({
                        k: v for k, v in new_data.items()
                        if k not in ["merge_with"]
                    })
                else:
                    # Just update existing location
                    if old_loc in self.memory["locations"]:
                        self.memory["locations"][old_loc].update({
                            k: v for k, v in new_data.items()
                            if k not in ["merge_with"]
                        })

            # Delete merged locations
            for loc in locations_to_delete:
                if loc in self.memory["locations"]:
                    del self.memory["locations"][loc]

            # Update people's location references
            for person_data in self.memory["people"].values():
                updated_locations = []
                for loc in person_data.get("locations", []):
                    # Check if this location was merged
                    canonical_found = None
                    for old_loc, new_data in consolidation_analysis["locations"].items():
                        if loc == old_loc and new_data.get("canonical_name") != old_loc:
                            canonical_found = new_data.get("canonical_name")
                            break

                    if canonical_found:
                        if canonical_found not in updated_locations:
                            updated_locations.append(canonical_found)
                    else:
                        if loc not in updated_locations:
                            updated_locations.append(loc)

                person_data["locations"] = updated_locations

        # Update relationships
        if "relationships" in consolidation_analysis:
            self.memory["relationships"] = consolidation_analysis["relationships"]

        # Update semantic facts
        if "semantic_facts" in consolidation_analysis:
            self.memory["semantic_facts"] = consolidation_analysis["semantic_facts"]

        # Update metadata
        self.memory["metadata"]["last_consolidation"] = datetime.now().isoformat()

        self._save_memory()