// Evidence gallery and image lightbox module, embedded by build_viz.py.
function escapeHtml(value){return String(value==null?'':value).replace(/[<>&"']/g,c=>({'<':'&lt;','>':'&gt;','&':'&amp;','"':'&quot;',"'":'&#39;'}[c]));}

function evidenceImageCaption(image){
  const at=Number(image.at_s||0).toFixed(2);
  return image.exact?'对应画面 · '+at+' 秒':image.requested_at_s!=null?'最近抽样画面 · '+at+' 秒（事实位置 '+Number(image.requested_at_s).toFixed(2)+' 秒）':'片段关键帧 · '+at+' 秒';
}
function openEntityEvidence(button){
  const dialog=document.getElementById('entityEvidenceDialog'),body=document.getElementById('entityEvidenceBody');
  if(!dialog||typeof dialog.showModal!=='function')return;
  let data={};
  try{data=JSON.parse(decodeURIComponent(button.dataset.evidence||''));}catch(_){return;}
  document.getElementById('entityEvidenceTitle').textContent=data.title||'记忆证据';
  const rows=[];
  if(data.segment)rows.push(['来源片段',data.segment]);
  if(data.video)rows.push(['源视频',data.video]);
  if(data.clipStart!=null)rows.push(['视频范围',Number(data.clipStart).toFixed(2)+'–'+Number(data.clipEnd??data.clipStart).toFixed(2)+' 秒']);
  if(data.times&&data.times.length)rows.push(['画面时间',data.times.map(time=>Number(time).toFixed(2)+' 秒').join('、')]);
  if(data.audioStart!=null)rows.push(['转写时间',Number(data.audioStart).toFixed(2)+'–'+Number(data.audioEnd??data.audioStart).toFixed(2)+' 秒']);
  const metadata=rows.length?'<dl class="entity-evidence-meta">'+rows.map(([label,value])=>'<dt>'+escapeHtml(label)+'</dt><dd>'+escapeHtml(value)+'</dd>').join('')+'</dl>':'';
  const text=data.evidence?'<div class="entity-evidence-text"><b>对应记忆内容</b><br>'+escapeHtml(data.evidence).replace(/\n/g,'<br>')+'</div>':'';
  const images=(Array.isArray(data.images)?data.images:[]).filter(image=>image&&image.url);
  let visual='';
  if(images.length){
    const first=images[0],caption=evidenceImageCaption(first);
    const thumbs=images.length>1?'<div class="entity-evidence-gallery-thumbs" aria-label="选择画面">'+images.map((image,index)=>'<button type="button" class="entity-evidence-gallery-thumb'+(index===0?' active':'')+'" data-gallery-index="'+index+'" aria-label="查看第 '+(index+1)+' 张画面" aria-pressed="'+(index===0?'true':'false')+'"><img loading="lazy" src="'+escapeHtml(image.url)+'" alt=""><span>'+(image.nightly_importance?'⭐ ':'')+Number(image.at_s||0).toFixed(2)+' 秒</span></button>').join('')+'</div>':'';
    visual='<div class="entity-evidence-frames entity-evidence-gallery" data-gallery-count="'+images.length+'">'
      +'<button type="button" class="entity-evidence-gallery-nav" data-gallery-move="-1" aria-label="上一张画面" '+(images.length>1?'':'hidden')+'>‹</button>'
      +'<figure class="entity-evidence-frame entity-evidence-current"><img loading="eager" src="'+escapeHtml(first.url)+'" '+(first.original?'data-original-url="'+escapeHtml(first.original)+'"':'')+' alt="来源画面 '+Number(first.at_s||0).toFixed(2)+' 秒"><span class="entity-evidence-highlight" '+(first.nightly_importance?'':'hidden')+' title="'+escapeHtml((first.nightly_importance||{}).reason||'夜间整理标记的重要画面')+'">⭐ 夜间重点</span><figcaption>'+caption+(first.original?' · 点击显示原始帧':'')+'</figcaption></figure>'
      +'<button type="button" class="entity-evidence-gallery-nav" data-gallery-move="1" aria-label="下一张画面" '+(images.length>1?'':'hidden')+'>›</button>'
      +'<div class="entity-evidence-gallery-status"><span class="entity-evidence-gallery-counter">1 / '+images.length+'</span><span>'+(images.length>1?'用 ← → 或缩略图切换':'关联画面预览')+'</span></div>'+thumbs+'</div>';
  }else{
    visual='<div class="entity-evidence-empty">这条记录暂时没有可用的关联画面。来源和时间仍可用于回查。</div>';
  }
  body.innerHTML=metadata+text+visual;
  const gallery=body.querySelector('.entity-evidence-gallery');
  if(gallery){
    let index=0;
    let showingOriginal=false;
    const captionSuffix=img=>img&&img.dataset.originalUrl
      ?(showingOriginal?' · 点击显示裁剪后的帧':' · 点击显示原始帧'):'';
    const setCaption=image=>{
      const cap=gallery.querySelector('.entity-evidence-current figcaption');
      if(cap)cap.textContent=evidenceImageCaption(image)+captionSuffix(gallery.querySelector('.entity-evidence-current img'));
    };
    gallery.querySelector('.entity-evidence-current').addEventListener('click',event=>{
      const img=event.target&&event.target.closest?event.target.closest('img'):null;
      if(!img||!img.dataset.originalUrl)return;
      event.stopPropagation();event.preventDefault();
      showingOriginal=!showingOriginal;
      img.src=showingOriginal?img.dataset.originalUrl:images[index].url;
      setCaption(images[index]);
    },true);
    const setImage=next=>{
      index=(next+images.length)%images.length;
      const image=images[index],preview=gallery.querySelector('.entity-evidence-current img');
      preview.src=image.url;
      preview.alt='来源画面 '+Number(image.at_s||0).toFixed(2)+' 秒';
      if(image.original)preview.dataset.originalUrl=image.original;else delete preview.dataset.originalUrl;
      showingOriginal=false;
      setCaption(image);
      const badge=gallery.querySelector('.entity-evidence-highlight');
      if(badge){badge.hidden=!image.nightly_importance;badge.title=(image.nightly_importance||{}).reason||'夜间整理标记的重要画面';}
      gallery.querySelector('.entity-evidence-gallery-counter').textContent=(index+1)+' / '+images.length;
      gallery.querySelectorAll('.entity-evidence-gallery-thumb').forEach((thumb,i)=>{
        thumb.classList.toggle('active',i===index);
        thumb.setAttribute('aria-pressed',i===index?'true':'false');
      });
    };
    gallery.querySelectorAll('[data-gallery-move]').forEach(control=>control.addEventListener('click',event=>{
      event.preventDefault();event.stopPropagation();setImage(index+Number(control.dataset.galleryMove||0));
    }));
    gallery.querySelectorAll('.entity-evidence-gallery-thumb').forEach((thumb,i)=>thumb.addEventListener('click',event=>{
      event.preventDefault();event.stopPropagation();setImage(i);
    }));
    gallery._setGalleryImage=setImage;
  }
  body.querySelectorAll('.entity-evidence-current img').forEach(image=>image.addEventListener('error',()=>{
    const empty=document.createElement('div');empty.className='entity-evidence-empty';empty.textContent='来源信息已保存，但对应画面文件当前不可用。';
    image.closest('.entity-evidence-gallery')?.replaceWith(empty);
  },{once:true}));
  body.querySelectorAll('.entity-evidence-gallery-thumb img').forEach(image=>image.addEventListener('error',()=>{
    image.closest('button')?.classList.add('unavailable');
  },{once:true}));
  if(!dialog.open)dialog.showModal();
  dialog.querySelector('.entity-evidence-close').focus();
}

(()=>{
  const dialog=document.getElementById('entityEvidenceDialog');
  if(!dialog)return;
  dialog.querySelector('.entity-evidence-close').addEventListener('click',()=>dialog.close());
  dialog.addEventListener('click',event=>{if(event.target===dialog)dialog.close();});
  dialog.addEventListener('keydown',event=>{
    if(event.key!=='ArrowLeft'&&event.key!=='ArrowRight')return;
    const gallery=dialog.querySelector('.entity-evidence-gallery');
    if(!gallery||Number(gallery.dataset.galleryCount||0)<2)return;
    event.preventDefault();gallery._setGalleryImage?.((Number(gallery.querySelector('.entity-evidence-gallery-counter').textContent.split('/')[0])-1)+(event.key==='ArrowRight'?1:-1));
  });
})();

// Bind each evidence album directly. Preload and decode the incoming page before
// animating, then swap the two image layers instead of replacing the visible image.
(()=>{
  const active=new Set();
  const readFrames=album=>{
    if(Array.isArray(album._albumFrames))return album._albumFrames;
    try{album._albumFrames=JSON.parse(decodeURIComponent(album.dataset.albumFrames||''));}
    catch(_){album._albumFrames=[];}
    return album._albumFrames;
  };
  const preload=(album,url)=>{
    if(!url)return Promise.resolve(null);
    album._albumImageCache=album._albumImageCache||new Map();
    if(album._albumImageCache.has(url))return album._albumImageCache.get(url);
    const promise=new Promise(resolve=>{
      const image=new Image();let settled=false;
      const done=async()=>{
        if(settled)return;settled=true;
        if(image.naturalWidth){try{await image.decode?.();}catch(_){} resolve(image);}
        else resolve(null);
      };
      image.decoding='async';image.onload=done;image.onerror=()=>{if(!settled){settled=true;resolve(null);}};
      image.src=url;
      if(image.complete)done();
    });
    album._albumImageCache.set(url,promise);
    return promise;
  };
  const detectOrientation=album=>{
    if(album.dataset.orientationLocked==='true')return;
    const cover=album.querySelector('.album-cover');
    if(!cover?.naturalWidth||!cover?.naturalHeight)return;
    const ratio=cover.naturalWidth/cover.naturalHeight;
    album.dataset.orientation=ratio<.82?'portrait':ratio>1.2?'landscape':'square';
    album.dataset.orientationLocked='true';
  };
  const advance=async(album,generation)=>{
    const frames=readFrames(album);
    if(frames.length<2||album._albumAnimating||album._albumAdvancing||generation!==album._albumGeneration)return;
    album._albumAdvancing=true;
    const current=Number(album.dataset.albumIndex||0)%frames.length;
    const next=(current+1)%frames.length,frame=frames[next];
    const ready=await preload(album,frame.url);
    if(!ready||generation!==album._albumGeneration||album.dataset.hoverCycling!=='true'){
      if(generation===album._albumGeneration){
        album._albumAdvancing=false;
        if(album.dataset.hoverCycling==='true')album._albumTimer=setTimeout(()=>advance(album,generation),450);
      }
      return;
    }
    const cover=album.querySelector('.album-cover'),turn=album.querySelector('.album-turn');
    if(!cover||!turn){album._albumAdvancing=false;return;}
    turn.classList.remove('is-turning');
    turn.src=frame.url;
    try{await turn.decode?.();}catch(_){}
    if(generation!==album._albumGeneration||album.dataset.hoverCycling!=='true')return;
    album._albumAdvancing=false;album._albumAnimating=true;
    // Both layers are decoded before the animation becomes visible. The current
    // cover stays in place; the next page rotates in above it, preventing a flash.
    void turn.offsetWidth;
    turn.classList.add('is-turning');
    album._albumFinishTimer=setTimeout(()=>{
      if(generation!==album._albumGeneration||album.dataset.hoverCycling!=='true')return;
      turn.classList.remove('album-turn','is-turning');
      turn.classList.add('album-cover');
      cover.classList.remove('album-cover');
      cover.classList.add('album-turn');
      album.dataset.albumIndex=String(next);
      const badge=album.querySelector('.memory-photo-album-highlight');
      if(badge){badge.hidden=!frame.important;badge.title=frame.importanceReason||'夜间整理标记的重要画面';}
      album._albumAnimating=false;album._albumFinishTimer=null;
      album._albumTimer=setTimeout(()=>advance(album,generation),520);
    },390);
  };
  const stop=album=>{
    album._albumGeneration=(album._albumGeneration||0)+1;
    if(album._albumDelay){clearTimeout(album._albumDelay);album._albumDelay=null;}
    if(album._albumTimer){clearTimeout(album._albumTimer);album._albumTimer=null;}
    if(album._albumFinishTimer){clearTimeout(album._albumFinishTimer);album._albumFinishTimer=null;}
    album._albumAnimating=false;album._albumAdvancing=false;
    album.querySelector('.album-turn')?.classList.remove('is-turning');
    album.classList.remove('is-cycling');delete album.dataset.hoverCycling;active.delete(album);
    album._albumImageCache?.clear();
  };
  const start=album=>{
    if(album._albumDelay||album._albumTimer||album._albumAnimating||album._albumAdvancing||readFrames(album).length<2)return;
    const generation=(album._albumGeneration||0)+1;album._albumGeneration=generation;
    active.add(album);album.classList.add('is-cycling');album.dataset.hoverCycling='true';
    album._albumDelay=setTimeout(()=>{
      album._albumDelay=null;
      advance(album,generation);
    },320);
  };
  const bind=album=>{
    if(!album||album.dataset.albumHoverBound)return;
    album.dataset.albumHoverBound='true';
    const cover=album.querySelector('.album-cover');
    cover?.addEventListener('load',()=>detectOrientation(album));
    detectOrientation(album);
    const first=readFrames(album)[0],badge=album.querySelector('.memory-photo-album-highlight');
    if(badge){badge.hidden=!first?.important;badge.title=first?.importanceReason||'夜间整理标记的重要画面';}
    album.addEventListener('pointerenter',()=>start(album));
    album.addEventListener('pointerleave',()=>stop(album));
    album.addEventListener('focusin',()=>start(album));
    album.addEventListener('focusout',event=>{if(!album.contains(event.relatedTarget))stop(album);});
  };
  const scan=root=>{
    if(root instanceof Element&&root.matches('.memory-photo-album'))bind(root);
    root.querySelectorAll?.('.memory-photo-album').forEach(bind);
  };
  const initialize=()=>{
    scan(document);
    if(document.body)new MutationObserver(records=>records.forEach(record=>record.addedNodes.forEach(node=>{
      if(node.nodeType===Node.ELEMENT_NODE)scan(node);
    }))).observe(document.body,{childList:true,subtree:true});
  };
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',initialize,{once:true});
  else initialize();
  document.addEventListener('visibilitychange',()=>{if(document.hidden)active.forEach(stop);});
})();
// Open evidence images at their full stored resolution; Escape, backdrop, or close button dismisses.
(()=>{
  const selector='.media-evidence img, .object-index-moment img, .entity-event img, .entity-evidence-dialog .entity-evidence-current img';
  let opener=null;
  const decorate=(root=document)=>{
    const images=[];
    if(root instanceof HTMLImageElement && root.matches(selector)) images.push(root);
    if(root.querySelectorAll) images.push(...root.querySelectorAll(selector));
    for(const img of images){
      if(img.closest('.memory-evidence-card-trigger')) continue;
      if(img.dataset.zoomReady) continue;
      img.dataset.zoomReady='1';
      img.classList.add('memory-image-zoomable');
      img.tabIndex=0;
      img.setAttribute('role','button');
      img.setAttribute('aria-label',(img.alt||'记忆证据图片')+'，点击放大');
      img.title='点击放大';
    }
  };
  const open=(img)=>{
    const dialog=document.getElementById('memoryImageLightbox');
    if(!dialog||typeof dialog.showModal!=='function') return;
    const preview=dialog.querySelector('img');
    const originalButton=dialog.querySelector('.memory-image-lightbox-original');
    opener=img;
    preview.src=img.currentSrc||img.src;
    preview.alt=img.alt||'记忆证据图片';
    preview.dataset.previewSrc=img.currentSrc||img.src;
    preview.dataset.originalSrc=img.dataset.originalUrl||'';
    preview.dataset.showingOriginal='false';
    originalButton.hidden=!preview.dataset.originalSrc||preview.dataset.originalSrc===preview.dataset.previewSrc;
    originalButton.textContent='查看原帧';
    if(!dialog.open) dialog.showModal();
    dialog.querySelector('.memory-image-lightbox-close').focus();
  };
  document.addEventListener('click',event=>{
    const card=event.target.closest?.('.memory-evidence-card-trigger');
    if(card){event.preventDefault();openEntityEvidence(card);return;}
    const img=event.target.closest?.(selector);
    if(img){event.preventDefault();open(img);}
  });
  document.addEventListener('keydown',event=>{
    const card=event.target.closest?.('.memory-evidence-card-trigger');
    if(card&&(event.key==='Enter'||event.key===' ')){event.preventDefault();openEntityEvidence(card);return;}
    const img=event.target.closest?.(selector);
    if(img&&(event.key==='Enter'||event.key===' ')){event.preventDefault();open(img);}
  });
  document.addEventListener('click',event=>{
    const dialog=document.getElementById('memoryImageLightbox');
    if(event.target===dialog) dialog.close();
  });
  const dialog=document.getElementById('memoryImageLightbox');
  if(dialog){
    dialog.querySelector('.memory-image-lightbox-close').addEventListener('click',()=>dialog.close());
    dialog.querySelector('.memory-image-lightbox-original').addEventListener('click',()=>{
      const preview=dialog.querySelector('img');
      const showingOriginal=preview.dataset.showingOriginal==='true';
      preview.src=showingOriginal?preview.dataset.previewSrc:preview.dataset.originalSrc;
      preview.dataset.showingOriginal=showingOriginal?'false':'true';
      dialog.querySelector('.memory-image-lightbox-original').textContent=showingOriginal?'查看原帧':'返回对应画面';
    });
    dialog.addEventListener('close',()=>{
      const preview=dialog.querySelector('img');
      preview.removeAttribute('src');
      delete preview.dataset.previewSrc;
      delete preview.dataset.originalSrc;
      delete preview.dataset.showingOriginal;
      const originalButton=dialog.querySelector('.memory-image-lightbox-original');
      originalButton.hidden=true;
      originalButton.textContent='查看原帧';
      if(opener?.isConnected) opener.focus();
      opener=null;
    });
  }
  decorate();
  new MutationObserver(records=>records.forEach(record=>record.addedNodes.forEach(node=>{
    if(node.nodeType===Node.ELEMENT_NODE) decorate(node);
  }))).observe(document.body,{childList:true,subtree:true});
})();
