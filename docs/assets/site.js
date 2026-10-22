(()=>{'use strict';const d=window.AVID,$=id=>document.getElementById(id),ns='http://www.w3.org/2000/svg';const node=(tag,attrs,txt)=>{let e=document.createElementNS(ns,tag);for(let [k,v] of Object.entries(attrs))e.setAttribute(k,v);if(txt!==undefined)e.textContent=txt;return e};const clear=e=>e.replaceChildren();const curve=(e,points,color)=>e.append(node('path',{d:points.map((p,i)=>(i?'L':'M')+p[0]+' '+p[1]).join(' '),stroke:color,'stroke-width':3,fill:'none','stroke-linecap':'round'}));const axes=(e,ytop,ybot,max,labels=[0,1,2,3,4,5],xleft=50,xright=615)=>{let x=t=>xleft+t/(labels.at(-1)-labels[0])*(xright-xleft),y=t=>ybot-t/max*(ybot-ytop);for(let i=0;i<=5;i++){let yy=y(max*i/5);e.append(node('line',{x1:xleft,x2:xright,y1:yy,y2:yy,stroke:'#91aab1','stroke-opacity':'.3'}));e.append(node('text',{x:xleft-9,y:yy+4,'text-anchor':'end',fill:'#77919a','font-size':11},(max*i/5).toFixed(2)))}for(let val of labels)e.append(node('text',{x:x(val),y:ybot+23,'text-anchor':'middle',fill:'#829ba1','font-size':11},String(val)));return {x,y}};
const rgb=$('rgb'), ir=$('ir'), status=$('video-status');
let scene='bird',desiredPlaying=true,switchToken=0;
const samples={bird:['BIRD_03897','IR_BIRD_03897'],drone:['DRONE_082192','IR_DRONE_082192']};
const videos=[rgb,ir];
const nameFor=t=>String(Math.floor(t/60)).padStart(2,'0')+':'+String(Math.floor(t%60)).padStart(2,'0');
const duration=()=>Number.isFinite(rgb.duration)&&rgb.duration>0?rgb.duration:0;
function message(text,isError=false){status.textContent=text;status.classList.toggle('error',isError)}
function updatePlayButton(){let playing=!rgb.paused&&!ir.paused;
 $('play').innerHTML=playing?'Ⅱ <span>PAUSE</span>':'▶ <span>PLAY</span>';
 $('play').setAttribute('aria-label',playing?'Pause paired videos':'Play paired videos');
}
function updateClock(){let t=rgb.currentTime||0,total=duration();
 $('scrub').value=total?Math.round(t/total*1000):0;
 $('timer').textContent=nameFor(t)+' / '+(total?nameFor(total):'—:—');
 if(ir.readyState>=2&&Math.abs((ir.currentTime||0)-t)>0.24){try{ir.currentTime=Math.min(t,Math.max(0,(ir.duration||total)-0.01))}catch(_){}}
}
async function setPlayback(play,{user=false}={}){
 desiredPlaying=play;
 if(!play){videos.forEach(v=>v.pause());message('Paused · use Play to resume');updatePlayButton();return;}
 if(videos.some(v=>v.error)){
  message('Video could not be decoded; the paired animated GIFs remain available below.',true);updatePlayButton();return;
 }
 let token=switchToken;
 try{
  const outcomes=await Promise.allSettled(videos.map(v=>v.play()));
  if(token!==switchToken)return;
  const failed=outcomes.find(x=>x.status==='rejected');
  if(failed){videos.forEach(v=>v.pause());
   let needsGesture=failed.reason&&failed.reason.name==='NotAllowedError';
   message(needsGesture?'Autoplay is blocked by this browser. Select Play to start both videos.':'Playback is unavailable in this viewer. Use the paired GIFs below.',!needsGesture);
  }else message('Playing both recordings · synchronized by frame time');
 }catch(err){message('Cannot start video playback. Use the paired GIFs below.',true)}
 updatePlayButton();updateClock();
}
$('play').onclick=()=>setPlayback(rgb.paused||ir.paused,{user:true});
function setSource(v,name){
 v.pause();
 v.poster='media/'+name+'.jpg';
 let sources=v.querySelectorAll('source');
 sources[0].src='media/'+name+'.mp4';
 if(sources[1])sources[1].src='media/'+name+'.webm';
 v.load();
}
document.querySelectorAll('[data-scene]').forEach(b=>b.onclick=()=>{
 scene=b.dataset.scene;switchToken++;
 document.querySelectorAll('[data-scene]').forEach(c=>c.setAttribute('aria-pressed',c===b?'true':'false'));
 $('sequence').textContent='SEQUENCE · '+(scene==='bird'?'BIRD_03897':'DRONE_082192');
 videos.forEach((v,i)=>setSource(v,samples[scene][i]));
 $('download-gif').href='media/'+(scene==='bird'?'BIRD':'DRONE')+'_RGB_IR.gif';
 $('scrub').value=0;message('Loading '+scene+' RGB–IR recordings…');
 // Wait for both clips to be seekable, then start the requested playback state.
 let token=switchToken;
 let starts=videos.map(v=>new Promise(resolve=>{
  if(v.readyState>=1)return resolve();
  const finish=()=>{v.removeEventListener('loadedmetadata',finish);v.removeEventListener('error',finish);resolve()};
  v.addEventListener('loadedmetadata',finish,{once:true});v.addEventListener('error',finish,{once:true});
 }));
 Promise.all(starts).then(()=>{if(token===switchToken){updateClock();if(desiredPlaying)setPlayback(true);else updatePlayButton()}});
});
document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>{
 document.querySelectorAll('[data-view]').forEach(c=>c.setAttribute('aria-pressed',b===c?'true':'false'));
 $('cameras').className='cameras'+(b.dataset.view==='both'?'':' only-'+b.dataset.view)
});
for(const v of videos){
 v.addEventListener('playing',()=>{updatePlayButton();if(!rgb.paused&&!ir.paused)message('Playing both recordings · synchronized by frame time')});
 v.addEventListener('pause',updatePlayButton);
 v.addEventListener('loadedmetadata',updateClock);
 v.addEventListener('error',()=>{
  const type=v.error?.code===4?'Unsupported codec or missing video file':'Media playback error';
  message(type+'. The animated paired GIFs are available below.',true);updatePlayButton();
 });
 v.addEventListener('stalled',()=>{if(desiredPlaying)message('Video is buffering… if playback does not resume, use the paired GIFs below.')});
}
rgb.addEventListener('timeupdate',updateClock);
$('scrub').oninput=()=>{
 let total=duration();if(!total)return;
 const t=Number($('scrub').value)/1000*total;
 for(let v of videos){if(v.readyState>=1){try{v.currentTime=Math.min(t,Math.max(0,(v.duration||total)-.01))}catch(_){}}}
 updateClock();
};
// An explicit play attempt allows the button to report blocked autoplay, rather than silently freezing at 0:00.
setPlayback(true);
const r=d.rho,N=r.length,count=t=>r.filter(v=>v>t).length;function drawRho(){let svg=$('rho-plot');clear(svg);let a=axes(svg,26,279,.6,[0,.25,.5,.75,1,1.25,1.5]);a.x=t=>50+t/1.5*565;let pts=Array.from({length:151},(_,i)=>[a.x(i/100),a.y(count(i/100)/N)]);curve(svg,pts,'#187b7b');let t=+$('rho-slider').value;svg.append(node('line',{x1:a.x(t),x2:a.x(t),y1:26,y2:279,stroke:'#d29b77','stroke-dasharray':'5 5'}));svg.append(node('circle',{cx:a.x(t),cy:a.y(count(t)/N),r:6,fill:'#d29b77'}));$('burden').textContent=(100*count(t)/N).toFixed(1)+'%';$('burden-count').textContent=count(t)+' of '+N+' annotated pairs';$('rho-label').textContent='ρ > '+t.toFixed(2)}$('rho-slider').oninput=drawRho;drawRho();let hist=$('histogram'),order=['0-4px','4-8px','8-12px','12-16px','16-20px','20-24px','24-28px','28-32px','32-48px','48-64px','64-96px','96-128px','128-infpx'];for(let k of order){let i=document.createElement('i');i.style.height=Math.max(1,d.size[k]/17706*100)+'%';if(k==='8-12px'||k==='12-16px')i.className='peak';i.title=k+': '+d.size[k].toLocaleString()+' annotated instances';hist.append(i)}
const fam={'gaussian_noise':'Gaussian noise','speckle_noise':'Speckle noise','salt_and_pepper':'Salt-and-pepper','gaussian_blur':'Gaussian blur','uneven_illumination':'Uneven illumination','motion_blur':'Motion blur','camera_instability':'Camera instability'};let active='gaussian_noise';for(let k in fam){let b=document.createElement('button');b.textContent=fam[k];b.setAttribute('aria-pressed',k===active?'true':'false');b.onclick=()=>{active=k;for(let c of $('families').children)c.setAttribute('aria-pressed',c===b?'true':'false');drawCorrupt()};$('families').append(b)}const bound=(m,s)=>d.corruption.find(x=>x.modality===m&&x.corruption_family===(s?' '+active:'clean').trim()&&x.severity===s)?.bayes_error_bound||0;function drawCorrupt(){let svg=$('corrupt-plot');clear(svg);let a=axes(svg,26,279,.5);let sev=+$('severity').value;for(let [m,col] of [['rgb','#c7835e'],['ir','#2f7e9e']]){let pts=Array.from({length:6},(_,i)=>[a.x(i),a.y(bound(m,i))]);curve(svg,pts,col);for(let i=0;i<6;i++)svg.append(node('circle',{cx:pts[i][0],cy:pts[i][1],r:sev===i?6:3,fill:col}))}svg.append(node('line',{x1:a.x(sev),x2:a.x(sev),y1:26,y2:279,stroke:'#809799','stroke-dasharray':'5 5'}));$('corrupt-name').textContent=fam[active];$('severity-label').textContent='LEVEL '+sev;let v=bound('rgb',sev),u=bound('ir',sev);$('rgb-bound').textContent=v.toFixed(3);$('ir-bound').textContent=u.toFixed(3);$('comparison').textContent=sev===0?'Clean input condition shown for reference.':v>u?'At this setting the RGB overlap bound is larger than the infrared bound.':'At this setting the infrared overlap bound is at least as large as the RGB bound.'}$('severity').oninput=drawCorrupt;drawCorrupt();
const f=d.fusion;function drawFusion(){let svg=$('fusion-plot');clear(svg);let max=f[0].bayes_fusion_gain;let x=i=>35+i/(f.length-1)*560,y=v=>180-(v-.7)/.4*148,pts=f.map((r,i)=>[x(i),y(r.bayes_fusion_gain/max)]);for(let i of [.8,1])svg.append(node('line',{x1:35,x2:595,y1:y(i),y2:y(i),stroke:'#5c7985','stroke-opacity':'.4'}));curve(svg,pts,'#9ce9d9');let idx=+$('fusion-slider').value;pts.forEach((p,i)=>{svg.append(node('circle',{cx:p[0],cy:p[1],r:idx===i?6:3,fill:idx===i?'#ffc28c':'#9ce9d9'}));svg.append(node('text',{x:p[0],y:204,'text-anchor':'middle','font-size':11,fill:'#a9c7d0'},String(f[i].injected_shift_px)))});$('fusion-label').textContent=f[idx].injected_shift_px+' px / axis';$('fusion-value').textContent=(f[idx].bayes_fusion_gain/max*100).toFixed(1)+'%'}$('fusion-slider').oninput=drawFusion;drawFusion();
})();
