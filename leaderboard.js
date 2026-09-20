(()=>{
  const data=window.DELIVERYGYM;
  const body=document.querySelector('#leaderboard-body');
  const metrics=['income','delivered','onTime','redLight','obstacle'];
  const render=mode=>{
    const rows=[...data.models.map(model=>({name:model.name,values:model[mode],type:'model'})),{name:data.oracle.name,values:data.oracle[mode],type:'oracle'}];
    const best=Math.max(...data.models.map(model=>model[mode][0]));
    body.innerHTML=rows.map(row=>`<tr class="${row.type==='oracle'?'oracle-row':''}"><th scope="row">${row.name}${row.type==='oracle'?`<span class="row-note">Simulator-assisted search reference</span>`:''}</th>${row.values.map((value,index)=>`<td class="${index===0?'income-cell':''}" ${index===0?`style="--income-width:${Math.max(8,value/24*100)}%;--income-color:${row.type==='oracle'?'#f7eedc':'#e6eef5'}"`:''}>${index===2?value.toFixed(1)+'%':value.toFixed(index===0?1:1)}</td>`).join('')}</tr>`).join('');
    document.querySelector('#best-income').textContent=`$${best.toFixed(1)}`;
    document.querySelectorAll('[data-mode]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.mode===mode)));
  };
  document.querySelector('#protocol').textContent=`${data.meta.protocol}. ${data.meta.aggregation}`;
  document.querySelectorAll('[data-mode]').forEach(button=>button.addEventListener('click',()=>render(button.dataset.mode)));
  const policies=data.policies.filter(policy=>['base','safety','uniform','adaptive'].includes(policy.id));
  document.querySelector('#trained-grid').innerHTML=policies.map(policy=>`<article class="${policy.id==='adaptive'?'featured':''}"><span>${policy.family==='reward'?'REWARD DESIGN':'CURRICULUM'}</span><h3>${policy.name}</h3><strong>$${policy.all.toFixed(2)}</strong><p>test net income</p><div><small>Familiar $${policy.familiar.toFixed(1)}</small><small>Unseen $${policy.unseen.toFixed(1)}</small></div></article>`).join('');
  render('waypoint');
})();
