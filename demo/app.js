"use strict";
const {studies, rows} = window.RAVET_DATA;
const $ = id => document.getElementById(id);
const fields = ["scenario", "dataset", "density", "retriever", "target", "expansion_kind", "expansion_factor"];
const names = {intact:"Intact", topic_subset:"Partial transfer", summary_only:"Compression", summary_topic:"Compression + transfer", sentinel_paraphrase:"Paraphrase", corpus_expansion:"Expansion", top1:"Top-1", top3:"Top-3", top5:"Top-5", top5_reference:"Top-5 reference", rechunk:"Re-chunking", rechunk_120_30:"Re-chunking", arguana:"ArguAna", scidocs:"SciDocs", bm25:"BM25", "bge-m3":"BGE-M3", "deepseek-v4-flash-0731":"DeepSeek-V4-Flash-0731", "kimi-k3":"Kimi-K3"};
const label = value => names[value] || value;
const pct = x => (100 * x).toFixed(1) + "%";
function options(select, values, all=true) {
  const previous = select.value;
  select.replaceChildren();
  if (all) select.add(new Option("All", "all"));
  values.forEach(v => select.add(new Option(label(v), v)));
  if ([...select.options].some(o=>o.value===previous)) select.value=previous;
}
Object.entries(studies).forEach(([key, value])=>$('study').add(new Option(value.label,key)));
let selection=[], current=[];
function rebuild() {
  const base=rows.filter(r=>r.study===$('study').value&&r.state===$('state').value);
  const expansion=$('study').value==='expansion';
  $('expansion-kind-label').hidden=!expansion;$('expansion-factor-label').hidden=!expansion;
  fields.forEach(k=>options($(k), [...new Set(base.map(r=>String(r[k])))].sort()));
  $('budget').max=studies[$('study').value].horizon;
  $('budget').value=Math.min(Number($('budget').value),Number($('budget').max));
  render();
}
function metric(rs,b){
  if (!rs.length) return {vsr:0,mq:0,n:0};
  return {vsr:rs.reduce((s,r)=>s+r.curve[b-1],0)/rs.length,mq:rs.reduce((s,r)=>s+b-r.curve.slice(0,b-1).reduce((a,x)=>a+x,0),0)/rs.length,n:rs.length};
}
function cell(row,text){const td=document.createElement('td');td.textContent=text;row.append(td);}
function render(){
  const study=studies[$('study').value], b=Number($('budget').value), clean=$('state').value==='clean';
  const base=rows.filter(r=>r.study===$('study').value&&r.state===$('state').value);
  fields.forEach(k=>[...$(k).options].forEach(o=>{o.disabled=o.value!=='all'&&!base.some(r=>String(r[k])===o.value&&fields.every(other=>other===k||$(other).value==='all'||String(r[other])===$(other).value));}));
  selection=rows.filter(r=>r.study===$('study').value&&r.state===$('state').value&&fields.every(k=>$(k).value==='all'||String(r[k])===$(k).value));
  current=['SentinelRAG','RAVET'].map(method=>({method,rows:selection.filter(r=>r.method===method)}));
  const a=metric(current[0].rows,b), c=metric(current[1].rows,b);
  $('success-title').textContent=clean?'False-positive rate':'Verification success';
  $('success-value').replaceChildren(document.createTextNode(pct(a.vsr)+' → '));
  const em=document.createElement('em');em.textContent=pct(c.vsr);$('success-value').append(em);
  const delta=100*(c.vsr-a.vsr);
  $('success-delta').textContent=(delta>=0?'+':'')+delta.toFixed(1)+' percentage points';
  $('success-delta').className=delta===0?'':((clean?delta<0:delta>0)?'positive':'negative');
  $('cost-value').replaceChildren(document.createTextNode(a.mq.toFixed(2)+' → '));
  const costEm=document.createElement('em');costEm.textContent=c.mq.toFixed(2);$('cost-value').append(costEm);
  const saving=a.mq?100*(a.mq-c.mq)/a.mq:0;
  $('cost-delta').textContent=Math.abs(saving).toFixed(1)+'% '+(saving>=0?'fewer':'more')+' capped queries';
  $('cost-delta').className=saving===0?'':saving>0?'positive':'negative';
  $('budget-value').textContent=b+' calls';$('budget-output').value=b;$('budget-max').textContent=study.horizon+' calls';
  $('bound').textContent='p₀ = '+study.p0.toFixed(4)+' · α = 0.01';
  $('sample-count').textContent=a.n+' paired configurations in selection';
  $('chart-title').textContent=clean?'Cumulative false-positive decisions':'Cumulative verification success';
  $('vsr-heading').textContent=(clean?'FPR':'VSR')+'('+b+') '+(clean?'↓':'↑');$('mq-heading').textContent='MQ('+b+') ↓';
  $('comparison').replaceChildren();
  current.forEach((item,i)=>{const tr=document.createElement('tr');if(i)tr.className='our-row';const m=metric(item.rows,b);[item.method+(i?' (Ours)':''),pct(m.vsr),m.mq.toFixed(2),String(m.n)].forEach(t=>cell(tr,t));$('comparison').append(tr);});
  $('scope-text').textContent=study.description+' Original archived bound; independent studies are not pooled. Ordering seeds are averaged within each configuration.';
  $('details').replaceChildren();
  const other=new Map(current[1].rows.map(r=>[r.condition_id,r]));
  current[0].rows.forEach(r=>{const rr=other.get(r.condition_id);if(!rr)return;const tr=document.createElement('tr');const am=metric([r],b),cm=metric([rr],b);[label(r.dataset)+' / '+label(r.scenario)+' / '+r.density+(r.study==='expansion'?' / '+r.expansion_kind+' '+r.expansion_factor+'x':''),label(r.retriever)+' / '+label(r.target)+' / '+r.replicate,pct(am.vsr)+' / '+am.mq.toFixed(2),pct(cm.vsr)+' / '+cm.mq.toFixed(2)].forEach(t=>cell(tr,t));$('details').append(tr);});
  draw();
}
function draw(){
  const canvas=$('chart'), rect=canvas.getBoundingClientRect(), ratio=window.devicePixelRatio||1;
  if(!rect.width)return;
  canvas.width=Math.round(rect.width*ratio);canvas.height=Math.round(rect.height*ratio);
  const ctx=canvas.getContext('2d');ctx.scale(ratio,ratio);
  const w=rect.width,h=rect.height,l=42,t=15,r=12,bt=35,max=Number($('budget').max),b=Number($('budget').value);
  const x=v=>l+v/max*(w-l-r), y=v=>h-bt-v*(h-t-bt);
  ctx.font='11px Arial';ctx.textAlign='right';ctx.textBaseline='middle';
  [0,.25,.5,.75,1].forEach(v=>{ctx.strokeStyle='#e6eaed';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(l,y(v));ctx.lineTo(w-r,y(v));ctx.stroke();ctx.fillStyle='#68717a';ctx.fillText(Math.round(v*100)+'%',l-8,y(v));});
  ctx.textAlign='center';[0,3,6,10,15,...(max===30?[20,25,30]:[])].forEach(v=>{ctx.fillText(v,x(v),h-17);});
  ctx.save();ctx.setLineDash([3,4]);ctx.strokeStyle='#b79154';ctx.beginPath();ctx.moveTo(x(b),t);ctx.lineTo(x(b),h-bt);ctx.stroke();ctx.restore();
  current.forEach((item,i)=>{if(!item.rows.length)return;ctx.strokeStyle=i?'#067969':'#636b78';ctx.lineWidth=i?2.8:2.1;ctx.setLineDash(i?[]:[6,4]);ctx.beginPath();ctx.moveTo(x(0),y(0));let last=0;for(let n=1;n<=max;n++){const v=metric(item.rows,n).vsr;ctx.lineTo(x(n),y(last));ctx.lineTo(x(n),y(v));last=v;}ctx.stroke();ctx.setLineDash([]);const v=metric(item.rows,b).vsr;ctx.beginPath();ctx.arc(x(b),y(v),4,0,Math.PI*2);ctx.fillStyle=i?'#067969':'white';ctx.fill();ctx.stroke();});
  canvas.setAttribute('aria-label',current.map(c=>{const m=metric(c.rows,b);return c.method+': '+pct(m.vsr)+' at '+b+' queries';}).join('; '));
}
$('study').addEventListener('change',()=>{fields.forEach(k=>$(k).value='all');rebuild();});fields.forEach(k=>$(k).addEventListener('change',render));$('state').addEventListener('change',rebuild);$('budget').addEventListener('input',render);
$('reset').addEventListener('click',()=>{$('study').value='primary';$('state').value='watermarked';fields.forEach(k=>$(k).value='all');$('budget').value=15;rebuild();});
['results','method'].forEach((tab,i)=>$(tab+'-tab').addEventListener('click',()=>{['results','method'].forEach((v,j)=>{$(v+'-tab').classList.toggle('active',i===j);$(v+'-tab').setAttribute('aria-selected',String(i===j));});$('evidence-view').hidden=!!i;$('method-view').hidden=!i;if(!i)draw();}));
$('download').addEventListener('click',()=>{const b=Number($('budget').value);const keys=['study','condition_id','method','scenario','dataset','density','retriever','target','replicate','state','expansion_kind','expansion_factor','p0'];const escape=x=>'"'+String(x).replaceAll('"','""')+'"';const lines=[[...keys,'budget','success','capped_queries'].join(',')];selection.forEach(r=>{const m=metric([r],b);lines.push([...keys.map(k=>r[k]),b,m.vsr,m.mq].map(escape).join(','));});const url=URL.createObjectURL(new Blob([lines.join('\n')+'\n'],{type:'text/csv'}));const a=document.createElement('a');a.href=url;a.download='ravet-selection.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),500);});
new ResizeObserver(draw).observe($('chart').parentElement);
rebuild();
