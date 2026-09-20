/* Authoritative website dataset. Values transcribed from active paper/main.tex
   tables and the labelled values in the figures included by that document.
   null means unreported; never substitute validation metrics for test metrics. */
window.DELIVERYGYM = {
  meta: {
    title: 'DeliveryGym: An RL Environment for Long-Horizon Embodied Agent Planning with Adaptive Curriculum',
    paper: 'https://arxiv.org/abs/2609.19801', code: 'https://github.com/mk322/DeliveryGym',
    authors: ['Haoqiang Kang', 'Yiming Zhang', 'Yiyang Guo', 'Chuying Li', 'Jianzhi Shen', 'Tianruo Rose Xu', 'Xiaokang Ye', 'Lianhui Qin'],
    year: 2026, youtubeId: '',
    checked: '2026-09-19',
    protocol: '130 test shifts · 13 city maps · 60-turn cap · 3 repeats per shift',
    aggregation: 'Average repeats within each shift, then weight shifts equally. On-time rate pools completed handoffs.',
    units: {income: 'USD / shift', delivered: 'orders / shift', onTime: '% of completed handoffs', redLight: 'events / shift', obstacle: 'events / shift'},
    sources: {
      benchmark: {file: 'paper/main.tex', label: 'tab:benchmark-main', split: 'test', interface: 'Waypoint and Any-Point'},
      selected: {file: 'paper/main.tex', label: 'tab:final-test-estimates', split: 'test', interface: 'Waypoint', note: 'Active measured table despite historical label suffix. Reward and curriculum policies are separate runs; selected steps 100 and 200 respectively.'},
      probes: {file: 'paper/figures/figure8_adaptive_reward.pdf', panel: 'B', split: 'training-only probes', unit: 'mean assertion score (%)', note: '16 fixed probes per skill; 20-turn cap; three assertions per probe; measured phase endpoints, not test success rates.'},
      scaling: {file: 'paper/figures/figure7_environment_scaling.pdf', panel: 'A / B', split: 'validation', unit: 'USD / shift', note: '2,400 trajectories and optimizer steps fixed within each sweep. Available task pool is not encountered unique tasks. Map sweep is not an unseen-city transfer test.'},
      diagnostics: {file: 'paper/figures/figure3_benchmark_diagnostics.pdf', split: 'test', interface: 'Waypoint', unit: '%', note: 'Reported rounded figure labels: single-order handoff success vs shift income / $24 search reference. Distinct protocols; not a causal decomposition.'}
    }
  },
  models: [
    {id:'gpt', name:'GPT-5.6 Sol', waypoint:[18.3,4.1,88.4,.5,1], anypoint:[9.2,1.7,58.2,1.7,3.3], diagnostic:[92,76]},
    {id:'claude', name:'Claude Fable 5', waypoint:[18.1,3.8,84.2,.7,1.2], anypoint:[8.6,1.5,54.4,2,3.7], diagnostic:[90,75]},
    {id:'kimi', name:'Kimi K3', waypoint:[16,3.2,78.3,1,1.7], anypoint:[6.2,1.1,46.1,2.4,4.3], diagnostic:[83,67]},
    {id:'qwen35', name:'Qwen3.6-35B-A3B', waypoint:[13.4,2.8,74.2,1.3,2], anypoint:[4.4,.8,39.5,2.8,4.9], diagnostic:[87,56]},
    {id:'glm', name:'GLM-5.3-Flash', waypoint:[14.7,3,76.3,1.2,1.9], anypoint:[5.3,1,42.8,2.6,4.6], diagnostic:[85,61]},
    {id:'qwen4', name:'Qwen3-VL-4B', waypoint:[7.6,.7,55.8,2.3,3.5], anypoint:[.8,.1,12.5,4,6.2], diagnostic:[65,32]}
  ],
  oracle: {name:'Oracle (search)', waypoint:[24,6.2,99,0,0], anypoint:[17.5,3.7,98.2,0,0], description:'Simulator-assisted search reference. It can access cloned simulator branches; it is not a guarantee of global optimality.'},
  cohorts: {all:{label:'All cities',n:130},familiar:{label:'Familiar cities',n:100},unseen:{label:'Unseen cities',n:30}},
  policies: [
    {id:'base',name:'Initialization',family:'reward',all:7.60,familiar:7.9,unseen:6.6,color:'neutral'},
    {id:'earnings',name:'Earnings-only RL',family:'reward',all:10.33,familiar:10.7,unseen:9.1,color:'rl'},
    {id:'safety',name:'Safety RL',family:'reward',all:11.73,familiar:12.1,unseen:10.5,color:'rl'},
    {id:'uniform',name:'Uniform',family:'curriculum',all:10.00,familiar:10.3,unseen:9,color:'neutral'},
    {id:'static',name:'Static',family:'curriculum',all:10.72,familiar:11,unseen:9.8,color:'muted'},
    {id:'random',name:'Random',family:'curriculum',all:10.40,familiar:10.7,unseen:9.4,color:'muted'},
    {id:'adaptive',name:'Adaptive',family:'curriculum',all:11.65,familiar:11.9,unseen:10.8,color:'adaptive'}
  ],
  probes: [
    {id:'traffic',name:'Traffic compliance',uniform:40,adaptive:56,block:4,explanation:'Practice routes through controlled intersections: cross safely, avoid red-light violations, and reach the destination.'},
    {id:'recovery',name:'Obstacle recovery',uniform:54,adaptive:72,block:8,explanation:'Practice feasible detours around blocked routes: use feedback to stop repeating the blocked edge and reach the destination.'},
    {id:'planning',name:'Multi-order planning',uniform:60,adaptive:86,block:12,explanation:'Practice overlapping orders: complete the required pair on time and earn at least 90% of the bounded planning reference.'}
  ],
  scaling: {
    tasks:{name:'Task scaling',unit:'available configurations',x:[200,800,3200,144000],income:[7.6,8.8,10.1,11.2],explanation:'A larger available task pool within one fixed training city. Pool size is not the number of unique tasks encountered.'},
    maps:{name:'Map scaling',unit:'training maps',x:[1,2,4,8],income:[9,10.8,12.7,14.5],explanation:'Nested training city sets broaden spatial experience. This validation sweep does not measure transfer to unseen cities.'}
  },
  figures: {
    learn:{name:'figure5_reward_design',caption:'Original paper figure · Validation training curves, not the selected-policy test values above.'},
    adapt:{name:'figure8_adaptive_reward',caption:'Original paper figure · (A) Validation training curves. (B) Training-only probe endpoints.'},
    scale:{name:'figure7_environment_scaling',caption:'Original paper figure · Validation income with a fixed interaction budget.'},
    beyond:{name:'figure3_benchmark_diagnostics',caption:'Original paper figure · Single-order success and shift income as percentages of their respective references.'}
  }
};
