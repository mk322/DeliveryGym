(() => {
  'use strict';
  const D = window.DELIVERYGYM;
  const $ = s => document.querySelector(s);
  const $$ = s => [...document.querySelectorAll(s)];
  const money = n => `$${n.toFixed(2)}`;
  const gain = (n, base) => (100 * (n / base - 1)).toFixed(1);
  const policy = id => D.policies.find(p => p.id === id);
  const colors = {neutral:'var(--neutral)',rl:'var(--rl)',adaptive:'var(--adaptive)',muted:'#7e8da5'};
  const state = {view:'learn',cohort:'all',adapt:'income',scale:'tasks',selected:'safety',mode:'waypoint'};
  const defaults = {learn:'safety',adapt:'adaptive',scale:'3',beyond:'gpt'};
  const stepText = [
    'Read street views, a route map, and structured state—including the resources and commitments left by earlier decisions.',
    'Choose orders, routes, deliveries, and resource actions; each tool call changes what remains possible later in the shift.',
    'Check executed simulator events for delivery outcomes, resource costs, traffic violations, and collisions to compute trajectory rewards.',
    'Use full-shift returns to update Qwen3-VL-4B with GRPO, learning from the consequences of its own actions.',
    'Measure weaknesses with fixed training-only probes, then adjust the distribution of future training tasks while keeping evaluation fixed.'
  ];
  $$('[data-step]').forEach(b => b.addEventListener('click', () => {
    $$('[data-step]').forEach(x => x.setAttribute('aria-pressed',String(x === b)));
    $('#step-explanation').textContent = stepText[Number(b.dataset.step)];
  }));
  $('#rl-headline').innerHTML = `+${gain(policy('safety').all,policy('base').all)}% <span>income with RL</span>`;
  $('#adaptive-headline').innerHTML = `+${gain(policy('adaptive').all,policy('uniform').all)}% <span>with adaptive curriculum</span>`;

  function options(items, selected) {
    $('#view-controls').innerHTML = items.map(([id,label]) => `<button data-option="${id}" aria-pressed="${id===selected}">${label}</button>`).join('');
    $$('[data-option]').forEach(b => b.addEventListener('click', () => {
      if (state.view === 'learn') state.cohort = b.dataset.option;
      if (state.view === 'adapt') {state.adapt = b.dataset.option;state.selected = state.adapt === 'income' ? 'adaptive' : state.adapt;}
      if (state.view === 'scale') {state.scale = b.dataset.option;state.selected = '3';}
      const selectedOption = b.dataset.option;
      renderExplorer();
      // Preserve keyboard position when the comparison controls are rebuilt.
      $$('[data-option]').find(option=>option.dataset.option===selectedOption)?.focus();
    }));
  }
  function bar(id, name, value, max, color, unit = 'money') {
    const text = unit === 'money' ? money(value) : `${value}%`;
    return `<button class="bar-row" data-select="${id}" aria-pressed="${state.selected===id}" aria-label="${name}: ${text}"><span class="bar-name">${name}</span><span class="bar-track" aria-hidden="true"><span class="bar-fill" style="--width:${value/max*100}%;--bar-color:${colors[color]}"></span></span><span class="bar-value">${text}</span></button>`;
  }
  function paired(id,name,first,second,firstColor,secondColor,labels) {
    return `<button class="bar-row paired-row" data-select="${id}" aria-pressed="${state.selected===id}" aria-label="${name}: ${labels[0]} ${first}%; ${labels[1]} ${second}%"><span class="bar-name">${name}</span><span class="paired-bars" aria-hidden="true"><span class="bar-track"><span class="bar-fill" style="--width:${first}%;--bar-color:${firstColor}"><span>${first}%</span></span></span><span class="bar-track"><span class="bar-fill" style="--width:${second}%;--bar-color:${secondColor}"><span>${second}%</span></span></span></span></button>`;
  }
  function legend(a,b,colorA,colorB) {
    return `<div class="legend"><span style="--legend-color:${colorA}">${a}</span><span style="--legend-color:${colorB}">${b}</span></div>`;
  }
  function renderExplorer() {
    const v = state.view;
    $('#results-panel').dataset.currentView = v;
    $('#results-panel').setAttribute('aria-labelledby',`tab-${v}`);
    const chart = $('#chart');
    const cohort = D.cohorts[state.cohort];
    if (v === 'learn') {
      $('#finding').textContent = 'Agents can learn to earn more from a complete shift.';
      options([['all','All'],['familiar','Familiar'],['unseen','Unseen']],state.cohort);
      $('#chart-label').textContent = `Test net income · USD / shift · ${cohort.label} (${cohort.n} shifts)`;
      chart.innerHTML = D.policies.filter(p=>p.family==='reward').map(p=>bar(p.id,p.name,p[state.cohort],14,p.color)).join('');
      $('#conditions').textContent = 'Qwen3-VL-4B · Waypoint · 60-turn cap. Safety RL adds traffic and obstacle penalties during training; test income excludes those extra reward penalties. Checkpoints selected on validation at step 100. Familiar: 100 new shifts on 10 training cities. Unseen: 30 shifts on 3 cities excluded from training and validation. Source: Appendix B, selected-policy test income.';
    } else if (v === 'adapt') {
      $('#finding').textContent = 'Practice is more useful when it follows the learner.';
      options([['income','Test income'],['traffic','Traffic'],['recovery','Recovery'],['planning','Multi-order']],state.adapt);
      if (state.adapt === 'income') {
        $('#chart-label').textContent = 'Test net income · USD / shift · All 130 shifts';
        chart.innerHTML = D.policies.filter(p=>p.family==='curriculum').map(p=>bar(p.id,p.name,p.all,14,p.color)).join('');
      } else {
        $('#chart-label').textContent = 'Training-only probes · Mean assertion score (%) · Measured phase endpoints';
        chart.innerHTML = legend('Adaptive','Uniform','var(--adaptive)','var(--neutral)') + D.probes.map(p=>paired(p.id,p.name,p.adaptive,p.uniform,'var(--adaptive)','var(--neutral)',['adaptive','uniform'])).join('');
      }
      $('#conditions').textContent = 'Qwen3-VL-4B · Waypoint. All four curricula share initialization, reward, optimizer steps, 2,400 training trajectories, and probe budgets. Static, random, and adaptive share 70% targeted practice. Test checkpoints: step 200 (separate from the reward study). Probes: 16 per skill, up to 20 turns; scores average three simulator assertions. Traffic / recovery / multi-order are measured at blocks 4 / 8 / 12. Sources: Appendix B and adaptive figure, panel B.';
    } else if (v === 'scale') {
      const s = D.scaling[state.scale];
      $('#finding').textContent = 'Broader experience helps within the same rollout budget.';
      options([['tasks','Task scaling'],['maps','Map scaling']],state.scale);
      $('#chart-label').textContent = 'Validation net income · USD / shift · Reported observations only';
      chart.innerHTML = `<div class="scale-plot">${[0,4,8,12,16].map(t=>`<span class="scale-tick" style="bottom:${t/16*100}%">$${t}</span>`).join('')}${s.x.map((x,i)=>`<button class="scale-point" style="--x:${i*30+5}%;--y:${s.income[i]/16*100}%" data-select="${i}" aria-pressed="${state.selected===String(i)}" aria-label="${x.toLocaleString('en-US')} ${s.unit}: ${money(s.income[i])} validation income"><span class="point-value">${money(s.income[i])}</span></button><span class="scale-x-label" style="--x:${i*30+5}%">${x.toLocaleString('en-US')}</span>`).join('')}</div><p class="scale-caption">${s.unit} · discrete conditions, equally spaced</p>`;
      $('#conditions').textContent = `Qwen3-VL-4B · Waypoint · Validation. Within each sweep, 2,400 sampled trajectories, optimizer steps, and evaluation maps are fixed. ${s.explanation} Relative gains are computed from the displayed means against the first condition. Source: environment-scaling figure, panel ${state.scale==='tasks'?'A':'B'}.`;
    } else {
      $('#finding').textContent = 'Delivering one order is only part of the challenge.';
      options([], '');
      $('#chart-label').textContent = 'Waypoint · Test · Percent of each protocol’s reference';
      chart.innerHTML = legend('Single-order success','Shift income / reference','var(--blue)','var(--oracle)')+D.models.map(m=>paired(m.id,m.name,...m.diagnostic,'var(--blue)','var(--oracle)',['single-order success','shift income / reference'])).join('');
      $('#conditions').textContent = 'Single-order protocol: one assigned feasible order per shift specification, verified handoff, 60-turn allowance. Shift protocol: autonomous work selection across the same 130 specifications. References: 100% single-order success and $24.00 Waypoint search income. Percentages reproduce rounded labels in the diagnostics figure; the two metrics have different denominators and are not a causal decomposition.';
    }
    const f = D.figures[v];
    $('#source-figure').open = false;
    $('#source-image').src = `assets/${f.name}.png`;
    $('#source-image').alt = f.caption;
    $('#source-zoom').dataset.zoom = `assets/${f.name}.png`;
    $('#source-zoom').setAttribute('aria-label','Enlarge original paper figure');
    $('#source-caption').innerHTML = `${f.caption} <a href="assets/${f.name}.pdf">View original PDF ↗</a>`;
    chart.querySelectorAll('[data-select]').forEach(b => {
      b.addEventListener('mouseenter',()=>readSelection(b.dataset.select,true));
      b.addEventListener('focus',()=>readSelection(b.dataset.select,true));
      b.addEventListener('mouseleave',()=>readSelection(state.selected));
      b.addEventListener('blur',()=>readSelection(state.selected));
      b.addEventListener('click',()=>{
        state.selected=b.dataset.select;
        if(state.view==='adapt' && state.adapt!=='income') {
          state.adapt=state.selected;
          $$('[data-option]').forEach(option=>option.setAttribute('aria-pressed',String(option.dataset.option===state.adapt)));
        }
        readSelection(state.selected);
      });
    });
    readSelection(state.selected);
  }
  function readSelection(id,preview=false) {
    let label,value,delta,explanation;
    const v = state.view;
    if (v === 'learn') {
      const p=policy(id), base=policy('base')[state.cohort], n=p[state.cohort];
      label=p.name;value=money(n);delta=id==='base'?'Untrained starting point':`+${money(n-base)} · +${gain(n,base)}% vs. initialization`;
      explanation=`${D.cohorts[state.cohort].label}: ${D.cohorts[state.cohort].n} fixed test shifts. ${id==='safety'?'Explicit safety penalties guide training; income is still measured from the economic ledger.':id==='earnings'?'The policy learns from simulator-verified shift earnings without extra safety penalties.':'The same small model before reinforcement learning.'}`;
    } else if (v === 'adapt' && state.adapt === 'income') {
      const p=policy(id),base=policy('uniform').all;
      label=p.name;value=money(p.all);delta=id==='uniform'?'Uniform-sampling reference':`+${money(p.all-base)} · +${gain(p.all,base)}% vs. uniform`;
      const descriptions={uniform:'Samples the base training distribution.',static:'Keeps targeted practice fixed to the initial weakness profile.',random:'Uses random targeting at each refresh.',adaptive:'Refreshes targeted practice using the policy’s measured weaknesses.'};
      explanation=`${descriptions[id]} These are separate curriculum runs, not an additional gain on top of safety RL.`;
    } else if (v === 'adapt') {
      const p=D.probes.find(p=>p.id===id);
      label=p.name;value=`${p.adaptive}%`;delta=`+${p.adaptive-p.uniform} percentage points vs. ${p.uniform}% uniform`;
      explanation=`${p.explanation} Training-probe endpoint: collection block ${p.block}.`;
    } else if (v === 'scale') {
      const s=D.scaling[state.scale],i=Number(id),n=s.income[i];
      label=`${s.x[i].toLocaleString('en-US')} ${s.unit}`;value=money(n);
      delta=i===0?'First condition in this sweep':`+${money(n-s.income[0])} · +${gain(n,s.income[0])}% vs. ${s.x[0].toLocaleString('en-US')}`;
      explanation=`Validation income with 2,400 sampled trajectories. ${s.explanation}`;
    } else {
      const m=D.models.find(m=>m.id===id);
      label=m.name;value=`${m.diagnostic[0]}% <small>/ ${m.diagnostic[1]}%</small>`;
      delta='Single-order success / shift income as % of reference';
      explanation=`${money(m.waypoint[0])} per shift vs. the $24.00 search reference. Completing an assigned delivery does not establish the ability to choose and coordinate profitable work across a shift.`;
    }
    $('#selection-label').textContent=label;$('#selection-value').innerHTML=value;$('#selection-delta').textContent=delta;$('#selection-explanation').textContent=explanation;
    $('#pin-state').textContent=preview?'Preview · click or Enter to pin':'Pinned selection';
    $$('[data-select]').forEach(b=>{b.setAttribute('aria-pressed',String(b.dataset.select===state.selected));b.classList.toggle('is-preview',preview && b.dataset.select===id);});
  }
  function selectView(view) {
    state.view=view;state.selected=defaults[view];
    if(view==='adapt') state.adapt='income';
    $$('[data-view]').forEach(b=>{const active=b.dataset.view===view;b.setAttribute('aria-selected',String(active));b.tabIndex=active?0:-1;});
    renderExplorer();
  }
  $$('[data-view]').forEach(b=>{
    b.addEventListener('click',()=>selectView(b.dataset.view));
    b.addEventListener('keydown',e=>{
      const tabs=$$('[data-view]'),i=tabs.indexOf(b);let next;
      if(e.key==='ArrowRight')next=(i+1)%tabs.length;
      if(e.key==='ArrowLeft')next=(i+tabs.length-1)%tabs.length;
      if(e.key==='Home')next=0;
      if(e.key==='End')next=tabs.length-1;
      if(next!==undefined){e.preventDefault();tabs[next].focus();selectView(tabs[next].dataset.view);}
    });
  });
  function renderTable() {
    const mode=state.mode, max=D.oracle[mode][0];
    const row=(name,values,kind='',note='')=>`<tr class="${kind}"><th scope="row">${name}${note?`<span class="row-note">${note}</span>`:''}</th>${values.map((n,i)=>`<td${i===0?` class="income-cell" style="--income-width:${n/max*100}%;--income-color:${kind==='oracle-row'?'#f2e8d1':kind==='adaptive-row'?'#eee6f6':kind==='training-row'?'#e1f0e8':'#e9eff5'}"`:''}>${n===null?'—':i===0?money(n):n.toFixed(1)}</td>`).join('')}</tr>`;
    const group=label=>`<tr class="group-row"><th colspan="6" scope="colgroup">${label}</th></tr>`;
    let rows=group('Zero-shot models')+D.models.map(m=>row(m.name,m[mode])).join('');
    if(mode==='waypoint') {
      rows+=group('Qwen3-VL-4B · Reward training');
      ['earnings','safety'].forEach(id=>{const p=policy(id);rows+=row(p.name,[p.all,null,null,null,null],'training-row');});
      rows+=group('Qwen3-VL-4B · Separate curriculum comparison');
      rows+=row('Adaptive curriculum',[policy('adaptive').all,null,null,null,null],'adaptive-row',`${money(policy('adaptive').all)} vs. ${money(policy('uniform').all)} uniform · same rollout budget`);
    }
    rows+=group('Simulator-assisted reference')+row(D.oracle.name,D.oracle[mode],'oracle-row','Search with cloned simulator branches');
    $('#benchmark-body').innerHTML=rows;
    $('#table-protocol').textContent=`${mode==='waypoint'?'Waypoint':'Any-Point'} / ${D.meta.protocol}`;
    $('#table-takeaway').innerHTML=mode==='waypoint'?`<span class="rl">${money(policy('base').all)} base → ${money(policy('safety').all)} safety RL.</span> Training improves the small model; frontier income (${money(D.models[0].waypoint[0])}) and the <span class="oracle">search reference (${money(max)})</span> still leave room to grow.`:`Finer movement makes the shift harder: frontier income is ${money(D.models[0].anypoint[0])}, against a ${money(max)} search reference. No Any-Point RL results are reported.`;
    $('#missing-note').hidden=mode!=='waypoint';
    $$('[data-mode]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.mode===mode)));
  }
  $$('[data-mode]').forEach(b=>b.addEventListener('click',()=>{state.mode=b.dataset.mode;renderTable();}));

  const dialog=$('#image-dialog');
  let opener;
  document.addEventListener('click',e=>{
    const b=e.target.closest('[data-zoom]');if(!b)return;
    opener=b;$('#dialog-image').src=b.dataset.zoom;$('#dialog-image').alt=b.querySelector('img').alt;$('#full-image').href=b.dataset.zoom;
    dialog.showModal();$('#close-dialog').focus();
  });
  $('#close-dialog').addEventListener('click',()=>dialog.close());
  dialog.addEventListener('click',e=>{if(e.target===dialog){const r=dialog.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)dialog.close();}});
  dialog.addEventListener('close',()=>opener?.focus());

  const bibtex=`@misc{kang2026deliverygym,
  title={${D.meta.title}},
  author={${D.meta.authors.join(' and ')}},
  year={2026},
  eprint={2609.19801},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={${D.meta.paper}}
}`;
  $('#bibtex').textContent=bibtex;
  $('#copy-citation').addEventListener('click',async()=>{
    try {
      if(!navigator.clipboard?.writeText)throw new Error('Clipboard unavailable');
      await navigator.clipboard.writeText(bibtex);
      $('#copy-feedback').textContent='BibTeX copied to clipboard.';
      $('#copy-citation').textContent='Copied ✓';
    } catch {
      const range=document.createRange();range.selectNodeContents($('#bibtex'));const selection=window.getSelection();selection.removeAllRanges();selection.addRange(range);
      $('#copy-feedback').textContent='Clipboard access is unavailable. The citation is selected; press Ctrl+C (or ⌘C) to copy.';
      $('#copy-citation').textContent='Select & copy';
    }
  });
  if(D.meta.youtubeId) {
    if(!/^[a-zA-Z0-9_-]{11}$/.test(D.meta.youtubeId))throw new Error('Expected an 11-character YouTube video ID');
    const iframe=document.createElement('iframe');iframe.src=`https://www.youtube-nocookie.com/embed/${D.meta.youtubeId}`;iframe.title='DeliveryGym in action';iframe.loading='lazy';iframe.allow='fullscreen; picture-in-picture';iframe.allowFullscreen=true;
    // Keep the labelled heading accessible when the placeholder is replaced.
    $('#video-container').innerHTML='<h2 id="video-title" class="sr-only">DeliveryGym in action</h2>';$('#video-container').append(iframe);
  }
  renderExplorer();renderTable();
})();
