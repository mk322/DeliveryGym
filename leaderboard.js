(()=>{
  const data=window.DELIVERYGYM;
  const body=document.querySelector('#leaderboard-body');
  const render=mode=>{
    const rows=[...data.models.map(model=>({name:model.name,values:model[mode],type:'model'})),{name:data.oracle.name,values:data.oracle[mode],type:'oracle'}];
    body.innerHTML=rows.map(row=>`<tr class="${row.type==='oracle'?'oracle-row':''}"><th scope="row">${row.name}</th>${row.values.map((value,index)=>`<td class="${index===0?'income-cell':''}" ${index===0?`style="--income-width:${Math.max(8,value/24*100)}%;--income-color:${row.type==='oracle'?'#f7eedc':'#e6eef5'}"`:''}>${index===2?value.toFixed(1)+'%':value.toFixed(1)}</td>`).join('')}</tr>`).join('');
    document.querySelector('#table-caption').textContent=`Zero-shot agent results with ${mode==='waypoint'?'Waypoint':'Any-Point'} movement`;
    document.querySelectorAll('[data-mode]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.mode===mode)));
  };
  document.querySelectorAll('[data-mode]').forEach(button=>button.addEventListener('click',()=>render(button.dataset.mode)));
  render('waypoint');
})();
