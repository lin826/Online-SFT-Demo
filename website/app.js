const byId = (id) => document.getElementById(id);

// Reading progress and active table of contents.
const progressBar = byId('reading-progress-bar');
const tocLinks = Array.from(document.querySelectorAll('.toc a'));
const trackedSections = tocLinks
  .map((link) => document.querySelector(link.getAttribute('href')))
  .filter(Boolean);
let scrollFrame = null;

function updateReadingState() {
  const root = document.documentElement;
  const scrollable = root.scrollHeight - root.clientHeight;
  const progress = scrollable > 0 ? Math.min(1, root.scrollTop / scrollable) : 0;
  progressBar.style.width = `${progress * 100}%`;

  let activeId = trackedSections[0]?.id;
  trackedSections.forEach((section) => {
    if (section.getBoundingClientRect().top <= 180) activeId = section.id;
  });
  tocLinks.forEach((link) => link.classList.toggle('active', link.getAttribute('href') === `#${activeId}`));
  scrollFrame = null;
}

window.addEventListener('scroll', () => {
  if (scrollFrame === null) scrollFrame = requestAnimationFrame(updateReadingState);
}, { passive: true });
updateReadingState();

// A deterministic pedagogical contextual-bandit sandbox.
const scenarios = [
  {
    title: 'Manager mentioned you', preview: '“Could you review the launch blocker?”', avatar: 'M', time: '09:14',
    category: 'manager', regime: 'weekday', importance: 0.88, deadline: 0.81, affinity: 0.46,
    latentNeed: 0.83, interruptionCost: 0.22,
  },
  {
    title: 'A friend shared a photo', preview: '“This reminded me of you.”', avatar: 'S', time: '17:12',
    category: 'social', regime: 'on-call', importance: 0.28, deadline: 0.10, affinity: 0.83,
    latentNeed: 0.38, interruptionCost: 0.74,
  },
  {
    title: 'Monitoring threshold crossed', preview: 'API latency is above the warning band.', avatar: '!', time: '02:08',
    category: 'monitoring', regime: 'off-hours', importance: 0.92, deadline: 0.96, affinity: 0.35,
    latentNeed: 0.94, interruptionCost: 0.58,
  },
  {
    title: 'Your receipt is ready', preview: 'Coffee order · $5.40', avatar: 'R', time: '13:43',
    category: 'receipt', regime: 'weekday', importance: 0.18, deadline: 0.05, affinity: 0.24,
    latentNeed: 0.12, interruptionCost: 0.34,
  },
  {
    title: 'Calendar starts soon', preview: 'Design review begins in 12 minutes.', avatar: 'C', time: '15:48',
    category: 'calendar', regime: 'weekday', importance: 0.76, deadline: 0.91, affinity: 0.61,
    latentNeed: 0.79, interruptionCost: 0.28,
  },
];

let scenarioIndex = 0;
let labCommitted = false;
const routeButtons = Array.from(document.querySelectorAll('[data-lab-action]'));
const labPending = byId('lab-pending');
const labReveal = byId('lab-reveal');

function clamp(value, low = 0, high = 1) {
  return Math.min(high, Math.max(low, value));
}

function formatProbability(value) {
  return `${Math.round(value * 100)}%`;
}

function evaluatorUtilities(scenario) {
  const urgency = 0.45 * scenario.importance + 0.55 * scenario.deadline;
  const relevance = 0.52 * scenario.affinity + 0.48 * scenario.latentNeed;
  return {
    interrupt: 0.82 * urgency + 0.34 * relevance - 0.72 * scenario.interruptionCost,
    later: 0.48 * scenario.importance + 0.42 * relevance - 0.31 * scenario.deadline + 0.12,
    archive: 0.46 * (1 - scenario.importance) + 0.25 * (1 - scenario.affinity) - 0.52 * scenario.deadline,
  };
}

function factualFeedback(scenario, action) {
  const engagement = 0.55 * scenario.affinity + 0.45 * scenario.latentNeed;
  if (action === 'interrupt') {
    if (engagement + scenario.deadline > 1.04) return ['Clicked after 8 seconds', 'The immediate notification was opened.'];
    if (scenario.interruptionCost > 0.58) return ['Dismissed immediately', 'The interruption was visible and rejected.'];
    return ['Ignored for 15 minutes', 'The push was delivered but not opened.'];
  }
  if (action === 'later') {
    if (engagement + scenario.importance > 0.92) return ['Opened in the digest', 'The delayed notification was opened with the digest.'];
    return ['Skipped in the digest', 'The digest appeared, but this item was not opened.'];
  }
  if (scenario.latentNeed > 0.72 && scenario.affinity > 0.45) return ['Organic app open', 'No push was sent; the user later opened the app independently.'];
  return ['No observable response', 'No push was sent and no organic open occurred.'];
}

function teacherDistribution(scenario, action, feedbackTitle) {
  const urgency = 0.5 * scenario.importance + 0.5 * scenario.deadline;
  const scores = {
    interrupt: 1.35 * urgency - 0.9 * scenario.interruptionCost,
    later: 0.65 * scenario.importance + 0.55 * scenario.affinity - 0.35 * scenario.deadline + 0.15,
    archive: 0.78 * (1 - scenario.importance) + 0.35 * (1 - scenario.affinity) - 0.58 * scenario.deadline,
  };
  if (feedbackTitle.startsWith('Clicked')) scores.interrupt += 0.38;
  if (feedbackTitle.startsWith('Dismissed')) scores.interrupt -= 0.45;
  if (feedbackTitle.startsWith('Opened in')) scores.later += 0.32;
  if (feedbackTitle.startsWith('Skipped')) scores.later -= 0.18;
  if (feedbackTitle.startsWith('Organic')) scores.archive += 0.24;
  if (feedbackTitle.startsWith('No observable')) scores[action] += 0.02;

  const temperature = 0.36;
  const maxScore = Math.max(...Object.values(scores));
  const exponentials = Object.fromEntries(Object.entries(scores).map(([key, value]) => [key, Math.exp((value - maxScore) / temperature)]));
  const total = Object.values(exponentials).reduce((sum, value) => sum + value, 0);
  return Object.fromEntries(Object.entries(exponentials).map(([key, value]) => [key, value / total]));
}

function renderScenario() {
  const scenario = scenarios[scenarioIndex];
  byId('lab-round').textContent = `#${scenarioIndex + 1}`;
  byId('lab-title').textContent = scenario.title;
  byId('lab-preview').textContent = scenario.preview;
  byId('lab-avatar').textContent = scenario.avatar;
  byId('lab-time').textContent = scenario.time;
  byId('lab-category').textContent = scenario.category;
  byId('lab-regime').textContent = scenario.regime;
  ['importance', 'deadline', 'affinity'].forEach((field) => {
    byId(`lab-${field}`).textContent = scenario[field].toFixed(2);
    byId(`${field}-bar`).style.width = `${scenario[field] * 100}%`;
  });
  labCommitted = false;
  labPending.hidden = false;
  labReveal.hidden = true;
  routeButtons.forEach((button) => {
    button.disabled = false;
    button.classList.remove('selected');
  });
}

function commitLabAction(action) {
  if (labCommitted) return;
  labCommitted = true;
  const scenario = scenarios[scenarioIndex];
  const [feedbackTitle, feedbackDetail] = factualFeedback(scenario, action);
  const distribution = teacherDistribution(scenario, action, feedbackTitle);
  const utilities = evaluatorUtilities(scenario);
  const regret = Math.max(...Object.values(utilities)) - utilities[action];

  routeButtons.forEach((button) => {
    button.disabled = true;
    button.classList.toggle('selected', button.dataset.labAction === action);
  });
  byId('feedback-title').textContent = feedbackTitle;
  byId('feedback-detail').textContent = `${feedbackDetail} The two alternate outcomes remain unobserved.`;
  byId('lab-regret').textContent = Math.max(0, regret).toFixed(2);
  const bestAction = Object.entries(utilities).sort((a, b) => b[1] - a[1])[0][0];
  byId('regret-detail').textContent = bestAction === action
    ? 'This route has the highest evaluator utility for the simulated round.'
    : `The evaluator preferred ${bestAction}; its utility vector never becomes a learning signal.`;
  const labels = { interrupt: 'A', later: 'B', archive: 'C' };
  byId('teacher-bars').innerHTML = Object.entries(distribution).map(([route, value]) => (
    `<div class="teacher-bar"><span>${labels[route]}</span><i><span style="width:${value * 100}%"></span></i><b>${formatProbability(value)}</b></div>`
  )).join('');
  labPending.hidden = true;
  labReveal.hidden = false;
}

routeButtons.forEach((button) => button.addEventListener('click', () => commitLabAction(button.dataset.labAction)));
byId('new-event').addEventListener('click', () => {
  scenarioIndex = (scenarioIndex + 1) % scenarios.length;
  renderScenario();
});
renderScenario();

// Step-through causal Online-SDFT round.
const roundSteps = [
  {
    phase: 'observe', title: '1. Observe public context',
    student: 'x<sub>t</sub><br><small>visible fields only</small>', world: '<b>sealed</b><small>waiting for an action</small>',
    explanation: 'The student receives only notification fields available at serving time.',
  },
  {
    phase: 'rollout', title: '2. Generate the student rollout',
    student: 'π(A,B,C | x<sub>t</sub>)<br><small>choose B · LATER</small>', world: '<b>sealed</b><small>no current feedback yet</small>',
    explanation: 'The LFM generates its own route without hidden user state z or a teacher demonstration.',
  },
  {
    phase: 'execute', title: '3. Execute exactly one route',
    student: 'a<sub>t</sub> = B<br><small>committed</small>', world: '<b>digest opened</b><small>only B produced an outcome</small>',
    explanation: 'Reality executes LATER. INTERRUPT and ARCHIVE outcomes are counterfactual and stay unknown.',
  },
  {
    phase: 'teach', title: '4. Ask the post-decision teacher',
    student: 'waiting<br><small>student cannot see z</small>', world: '<b>q = [.12, .81, .07]</b><small>x, action, z, factual feedback</small>',
    explanation: 'The teacher returns a soft route distribution using legal post-decision evidence—not the evaluator’s oracle utilities.',
  },
  {
    phase: 'update', title: '5. Update a tiny adapter',
    student: 'LoRA ← q<sub>t</sub><br><small>172,032 parameters</small>', world: '<b>round complete</b><small>advance to xₜ₊₁</small>',
    explanation: 'One small online update prepares the student for the next notification in the stream.',
  },
];
let roundStep = 0;
let roundTimer = null;
const roundStage = document.querySelector('.round-stage');
const stepButtons = Array.from(document.querySelectorAll('[data-round-step]'));
const roundPlay = byId('round-play');

function renderRoundStep(index) {
  roundStep = index;
  const step = roundSteps[index];
  roundStage.dataset.phase = step.phase;
  byId('step-title').textContent = step.title;
  byId('student-screen').innerHTML = step.student;
  byId('world-screen').innerHTML = step.world;
  byId('step-explanation').textContent = step.explanation;
  stepButtons.forEach((button, buttonIndex) => {
    if (buttonIndex === index) button.setAttribute('aria-current', 'step');
    else button.removeAttribute('aria-current');
  });
}

function stopRoundPlayback() {
  if (roundTimer !== null) window.clearInterval(roundTimer);
  roundTimer = null;
  roundPlay.innerHTML = '<span aria-hidden="true">▶</span> Play';
  roundPlay.setAttribute('aria-label', 'Play the online round');
}

function toggleRoundPlayback() {
  if (roundTimer !== null) {
    stopRoundPlayback();
    return;
  }
  roundPlay.innerHTML = '<span aria-hidden="true">Ⅱ</span> Pause';
  roundPlay.setAttribute('aria-label', 'Pause the online round');
  renderRoundStep((roundStep + 1) % roundSteps.length);
  roundTimer = window.setInterval(() => renderRoundStep((roundStep + 1) % roundSteps.length), 1900);
}

stepButtons.forEach((button) => button.addEventListener('click', () => {
  stopRoundPlayback();
  renderRoundStep(Number(button.dataset.roundStep));
}));
roundPlay.addEventListener('click', toggleRoundPlayback);
renderRoundStep(0);

// Interactive comparison of the six methods.
const MAX_REGRET = 115.65;
const methods = {
  base: {
    name: 'Base', family: 'Frozen weights · no memory', score: 37.08, regret: 81.50,
    description: 'Serves every notification with the pretrained Liquid LFM and never adapts to the user or stream.',
    input: 'Visible context only', signal: 'None', change: 'Nothing',
    flow: {
      serve: [
        { label: 'Visible context x', state: 'used' },
        { label: 'Hidden state z', state: 'sealed' },
        { label: 'Evaluator utilities', state: 'sealed' },
        { label: 'Prompt memory', state: 'absent' },
      ],
      post: [
        { label: 'Factual route outcome', state: 'observed' },
        { label: 'Scalar reward', state: 'absent' },
        { label: 'Teacher hard sample', state: 'absent' },
        { label: 'Teacher soft distribution', state: 'absent' },
      ],
      adapt: [
        { label: 'No parameter update', state: 'absent' },
        { label: 'No cross-round memory', state: 'absent' },
      ],
    },
  },
  icl: {
    name: 'ICL', family: 'Frozen weights · recent memory', score: 37.50, regret: 81.10,
    description: 'Adds the 12 most recent legal teacher-labeled examples to the prompt. The LFM weights remain frozen.',
    input: 'Context + 12 recent examples', signal: 'Past sampled teacher routes', change: 'Prompt only',
    flow: {
      serve: [
        { label: 'Visible context x', state: 'used' },
        { label: 'Hidden state z', state: 'sealed' },
        { label: 'Evaluator utilities', state: 'sealed' },
        { label: 'Last 12 teacher examples', state: 'used' },
      ],
      post: [
        { label: 'Factual route outcome', state: 'observed' },
        { label: 'Scalar reward', state: 'absent' },
        { label: 'Teacher hard sample', state: 'used' },
        { label: 'Teacher soft distribution', state: 'absent' },
      ],
      adapt: [
        { label: 'Weights stay frozen', state: 'absent' },
        { label: 'Append example after round', state: 'used' },
      ],
    },
  },
  rag: {
    name: 'RAG', family: 'Frozen weights · retrieved memory', score: 38.75, regret: 79.94,
    description: 'Retrieves 12 similar past contexts with a mixed-feature distance and places the strongest match next to the current query.',
    input: 'Context + 12 nearest examples', signal: 'Past sampled teacher routes', change: 'Prompt only',
    flow: {
      serve: [
        { label: 'Visible context x', state: 'used' },
        { label: 'Hidden state z', state: 'sealed' },
        { label: 'Evaluator utilities', state: 'sealed' },
        { label: '12 nearest past contexts', state: 'used' },
      ],
      post: [
        { label: 'Factual route outcome', state: 'observed' },
        { label: 'Scalar reward', state: 'absent' },
        { label: 'Teacher hard sample', state: 'used' },
        { label: 'Teacher soft distribution', state: 'absent' },
      ],
      adapt: [
        { label: 'Weights stay frozen', state: 'absent' },
        { label: 'Index record after round', state: 'used' },
      ],
    },
  },
  reinforce: {
    name: 'REINFORCE', family: 'Weights adapt online', score: 32.08, regret: 115.65,
    description: 'Samples one on-policy action and updates LoRA from its scalar factual reward and a past-only EMA baseline. It never queries the teacher.',
    input: 'Visible context only', signal: 'One factual scalar reward', change: '172,032 LoRA parameters',
    flow: {
      serve: [
        { label: 'Visible context x', state: 'used' },
        { label: 'Hidden state z', state: 'sealed' },
        { label: 'Evaluator utilities', state: 'sealed' },
        { label: 'Prompt memory', state: 'absent' },
      ],
      post: [
        { label: 'Factual route outcome', state: 'observed' },
        { label: 'Scalar reward', state: 'used' },
        { label: 'Teacher hard sample', state: 'absent' },
        { label: 'Teacher soft distribution', state: 'absent' },
      ],
      adapt: [
        { label: 'LoRA policy-gradient step', state: 'used' },
        { label: 'Past-only EMA baseline', state: 'used' },
      ],
    },
  },
  sft: {
    name: 'Online-SFT', family: 'Weights adapt online', score: 41.94, regret: 97.65,
    description: 'Samples one teacher route after execution, converts it to a hard one-hot target, and updates the LoRA adapter online.',
    input: 'Visible context only', signal: 'One sampled teacher route', change: '172,032 LoRA parameters',
    flow: {
      serve: [
        { label: 'Visible context x', state: 'used' },
        { label: 'Hidden state z', state: 'sealed' },
        { label: 'Evaluator utilities', state: 'sealed' },
        { label: 'Prompt memory', state: 'absent' },
      ],
      post: [
        { label: 'Factual route outcome', state: 'observed' },
        { label: 'Scalar reward', state: 'absent' },
        { label: 'Teacher hard sample', state: 'used' },
        { label: 'Teacher soft distribution', state: 'absent' },
      ],
      adapt: [
        { label: 'LoRA one-hot update', state: 'used' },
        { label: 'Online replay batch of 4', state: 'used' },
      ],
    },
  },
  sdft: {
    name: 'Online-SDFT', family: 'Weights adapt online', score: 64.72, regret: 36.24,
    description: 'Distills the teacher’s complete soft action distribution into a rank-4 LoRA adapter after each executed action.',
    input: 'Visible context only', signal: 'Post-decision teacher distribution', change: '172,032 LoRA parameters',
    flow: {
      serve: [
        { label: 'Visible context x', state: 'used' },
        { label: 'Hidden state z', state: 'sealed' },
        { label: 'Evaluator utilities', state: 'sealed' },
        { label: 'Prompt memory', state: 'absent' },
      ],
      post: [
        { label: 'Factual route outcome', state: 'observed' },
        { label: 'Scalar reward', state: 'absent' },
        { label: 'Teacher hard sample', state: 'absent' },
        { label: 'Teacher soft distribution', state: 'used' },
      ],
      adapt: [
        { label: 'LoRA soft distillation', state: 'used' },
        { label: 'Online replay batch of 4', state: 'used' },
      ],
    },
  },
};
const methodTabs = Array.from(document.querySelectorAll('[data-method]'));
const signalLists = {
  serve: byId('signal-serve'),
  post: byId('signal-post'),
  adapt: byId('signal-adapt'),
};
const stateLabels = {
  used: 'Used for learning',
  observed: 'Observed; not a learning target',
  sealed: 'Sealed from the student',
  absent: 'Not used by this method',
};

function renderSignalList(listElement, items) {
  listElement.innerHTML = items.map((item) => (
    `<li class="signal-item signal-${item.state}">
      <span class="signal-state" aria-hidden="true"></span>
      <span class="signal-label">${item.label}</span>
      <span class="sr-only">${stateLabels[item.state]}</span>
    </li>`
  )).join('');
}

function activateMethod(methodKey, focus = false) {
  const method = methods[methodKey];
  byId('method-name').textContent = method.name;
  byId('method-family').textContent = method.family;
  byId('method-description').textContent = method.description;
  byId('method-input').textContent = method.input;
  byId('method-signal').textContent = method.signal;
  byId('method-change').textContent = method.change;
  byId('method-score').textContent = `${method.score.toFixed(2)}%`;
  byId('method-score-bar').style.width = `${method.score}%`;
  byId('method-regret').textContent = method.regret.toFixed(2);
  byId('method-regret-bar').style.width = `${(method.regret / MAX_REGRET) * 100}%`;
  renderSignalList(signalLists.serve, method.flow.serve);
  renderSignalList(signalLists.post, method.flow.post);
  renderSignalList(signalLists.adapt, method.flow.adapt);
  byId('signal-flow-caption').textContent = (
    `${method.name}: information channels at serving time, after the executed action, and during adaptation. `
    + 'Sealed channels never enter the student prompt, replay, teacher target, reward, or gradient.'
  );
  methodTabs.forEach((tab) => {
    const selected = tab.dataset.method === methodKey;
    tab.setAttribute('aria-selected', String(selected));
    tab.tabIndex = selected ? 0 : -1;
  });
  const selectedTab = methodTabs.find((tab) => tab.dataset.method === methodKey);
  byId('method-panel').setAttribute('aria-labelledby', selectedTab.id);
  if (focus) selectedTab.focus();
}

methodTabs.forEach((tab, index) => {
  tab.addEventListener('click', () => activateMethod(tab.dataset.method));
  tab.addEventListener('keydown', (event) => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    let nextIndex = index;
    if (event.key === 'ArrowRight') nextIndex = (index + 1) % methodTabs.length;
    if (event.key === 'ArrowLeft') nextIndex = (index - 1 + methodTabs.length) % methodTabs.length;
    if (event.key === 'Home') nextIndex = 0;
    if (event.key === 'End') nextIndex = methodTabs.length - 1;
    activateMethod(methodTabs[nextIndex].dataset.method, true);
  });
});
activateMethod('sdft');
