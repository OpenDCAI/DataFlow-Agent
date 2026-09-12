(() => {
  'use strict';
  const payload = JSON.parse(document.getElementById('trajectory-data').textContent);
  const $ = id => document.getElementById(id);
  const list = value => Array.isArray(value) ? value : [];
  const obj = value => value && typeof value === 'object' && !Array.isArray(value) ? value : {};
  const text = value => value == null ? '—' : typeof value === 'string' ? value : JSON.stringify(value, null, 2);
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const fmt = value => finite(value) ? value.toLocaleString('en-US', {maximumFractionDigits: 4}) : '—';
  const validIndex = (value, items) => Number.isInteger(value) && value >= 0 && value < items.length;
  const el = (tag, className, value) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (value !== undefined) node.textContent = text(value);
    return node;
  };
  const badge = (label, kind = '') => el('span', 'badge ' + kind, label);
  const note = (value, kind = '') => el('div', 'notice ' + kind, value);
  const json = value => el('pre', '', text(value));
  function details(label, content, open = false) {
    const node = el('details'); node.open = open;
    node.append(el('summary', '', label));
    const body = el('div', 'detail-body');
    if (typeof content === 'function') {
      let loaded = false;
      const load = () => { if (node.open && !loaded) { loaded = true; content(body); } };
      node.addEventListener('toggle', load);
      if (open) load();
    } else if (content) body.append(content);
    node.append(body); return node;
  }
  function dialog(title, render) {
    $('dialog-title').textContent = title;
    $('dialog-content').replaceChildren(); render($('dialog-content'));
    if (!$('detail-dialog').open) $('detail-dialog').showModal();
  }
  $('dialog-close').addEventListener('click', () => $('detail-dialog').close());
  function imageFigure(block) {
    const asset = obj(payload.assets)[block.asset_id];
    if (!asset) return note('图片资源缺失。', 'warn');
    const figure = el('figure', 'image-figure');
    const button = el('button', 'image-button'); button.type = 'button';
    button.setAttribute('aria-label', '放大 observation 图片');
    const image = el('img'); image.loading = 'lazy'; image.decoding = 'async';
    image.alt = '轨迹中记录的图片 observation';
    image.src = 'data:' + asset.media_type + ';base64,' + asset.data;
    image.addEventListener('error', () => figure.replaceChildren(note('图片无法解码。', 'warn')));
    button.append(image);
    button.addEventListener('click', () => dialog('图片 observation · 原始内嵌分辨率', body => {
      const full = image.cloneNode(); full.loading = 'eager'; full.className = 'full-image'; body.append(full);
    }));
    figure.append(button, el('figcaption', '', asset.media_type + ' · 点击放大 · detail=' + text(block.detail)));
    return figure;
  }
  function content(parent, blocks) {
    if (typeof blocks === 'string') { parent.append(el('div', 'text-block', blocks)); return; }
    if (!Array.isArray(blocks)) { parent.append(note('缺少规范 content 数组。', 'warn'), json(blocks)); return; }
    let grid = null;
    for (const raw of blocks) {
      const block = obj(raw);
      if (block.type === 'image' && block.asset_id) {
        if (!grid) { grid = el('div', 'image-grid'); parent.append(grid); }
        grid.append(imageFigure(block)); continue;
      }
      grid = null;
      if (block.type === 'text') {
        let parsed;
        try { parsed = JSON.parse(block.text); } catch (_) { /* plain text */ }
        parent.append(parsed && typeof parsed === 'object' ? json(parsed) : el('div', 'text-block', block.text));
      } else if (block.type === 'unavailable_image') {
        parent.append(note('图片未内嵌：' + text(block.reason), 'warn'));
      } else {
        parent.append(note('未支持的内容块，仅展示记录；不会访问其中的 URL 或路径。', 'warn'), json(raw));
      }
    }
  }
  function message(raw, index) {
    const value = obj(raw), node = el('article', 'message');
    const head = el('div', 'message-head');
    head.append(el('span', 'message-index', '#' + index), badge(value.role || 'unknown', 'role'));
    if (value.name) head.append(el('span', '', value.name));
    node.append(head); content(node, value.content); return node;
  }
  function messages(parent, items, indexes = items.map((_, i) => i)) {
    if (!indexes.length) parent.append(el('p', 'empty', '没有记录。'));
    for (const index of indexes) parent.append(message(items[index], index));
  }
  function elapsed(trajectory) {
    const start = Date.parse(trajectory.started_at), end = Date.parse(trajectory.completed_at);
    return Number.isFinite(start) && Number.isFinite(end) && end >= start ? fmt((end - start) / 1000) + ' s' : '未记录';
  }
  function status(trajectory) {
    const reason = trajectory.termination_reason || '未记录终止状态';
    const good = ['finish', 'environment_final'].includes(reason);
    return badge(reason, good ? 'good' : reason === 'infrastructure_error' ? 'bad' : 'warn');
  }
  const records = list(payload.records);
  let selectedRecord = 0, selectedVariant = 0;
  document.title = text(payload.title);
  $('report-title').textContent = text(payload.title);

  function recordList() {
    const query = $('record-search').value.trim().toLowerCase();
    $('record-list').replaceChildren(); let count = 0;
    records.forEach((record, index) => {
      const haystack = [record.task_id, record.env_id, ...record.variants.map(v => v.trajectory.episode_id)].join(' ').toLowerCase();
      if (query && !haystack.includes(query)) return;
      count++;
      const button = el('button', 'record-button'); button.type = 'button';
      button.setAttribute('aria-current', String(index === selectedRecord));
      button.append(el('div', 'record-name', record.task_id), el('small', '', record.env_id + ' · 记录 ' + (index + 1)));
      button.addEventListener('click', () => {
        selectedRecord = index; selectedVariant = 0; render(); window.scrollTo(0, 0);
      });
      $('record-list').append(button);
    });
    $('record-count').textContent = count + ' / ' + records.length + ' 条记录';
    if (!count) $('record-list').append(el('p', '', '没有匹配的记录。'));
  }
  $('record-search').addEventListener('input', recordList);

  function renderTask(record, trajectory) {
    const parent = $('task-content'), history = list(trajectory.messages), steps = list(trajectory.steps);
    const first = obj(steps[0]).response_message_index;
    const boundary = validIndex(first, history) ? first : history.length;
    parent.append(note('以下来自 trajectory 的规范消息记录，不是服务商网络原始请求。Serving 可能做角色转换、图片截断或上下文压缩；这些转换未记录时无法还原。'));
    const userIndexes = history.map((m, i) => i).filter(i => i < boundary && obj(history[i]).role === 'user');
    const panel = el('div', 'panel');
    panel.append(el('h3', '', '首次有记录的模型决策前 · 用户任务与补充指令'));
    messages(panel, history, userIndexes); parent.append(panel);
    if (!validIndex(first, history)) parent.append(note('未找到有效的首步响应索引：仅按消息角色展示现有记录，不证明模型已经收到这些输入。', 'warn'));
    const systemIndexes = history.map((m, i) => i).filter(i => obj(history[i]).role === 'system');
    parent.append(details('System prompt · ' + systemIndexes.length + ' 条（完整记录，含工具目录）', body => messages(body, history, systemIndexes)));
    parent.append(details('完整初始上下文 · message #0–' + Math.max(0, boundary - 1), body => messages(body, history.slice(0, boundary))));
    if (record.task) parent.append(details('补充 Task 定义 · 仅公开字段，不含 Scenario / verifier', json(record.task)));
  }

  function renderTimeline(trajectory) {
    const parent = $('timeline'), history = list(trajectory.messages), steps = list(trajectory.steps);
    $('step-count').textContent = steps.length + ' STEPS';
    if (!steps.length) { parent.append(el('p', 'empty', '没有 step 记录。初始失败信息仍可在“全部消息”中查看。')); return; }
    steps.forEach((raw, ordinal) => {
      const step = obj(raw), action = obj(step.action);
      const card = el('article', 'step' + (step.tool_ok === false ? ' failed' : ''));
      card.dataset.step = String(ordinal + 1);
      const head = el('div', 'step-header');
      head.append(el('span', 'step-number', String(step.index ?? ordinal + 1).padStart(2, '0')),
                  el('span', 'tool-name', action.tool || (step.parse_error ? 'parse error' : '未知 action')),
                  badge(step.tool_ok === true ? 'OK' : step.tool_ok === false ? 'ERROR' : 'UNKNOWN', step.tool_ok === true ? 'good' : step.tool_ok === false ? 'bad' : ''));
      if (step.is_final) head.append(badge('FINAL'));
      if (finite(step.elapsed_ms)) head.append(el('span', 'step-time', fmt(step.elapsed_ms) + ' ms · 记录耗时'));
      const input = el('button', 'step-input', '查看该步输入'); input.type = 'button';
      const responseIndex = step.response_message_index;
      input.disabled = !validIndex(responseIndex, history);
      input.addEventListener('click', () => dialog('Step ' + (ordinal + 1) + ' · 记录的模型输入前缀', body => {
        body.append(note('显示该响应之前的规范消息前缀；不声称等同于实际 provider 请求。'));
        messages(body, history.slice(0, responseIndex));
      }));
      head.append(input); card.append(head);
      const body = el('div', 'step-body');
      if (step.error_code) body.append(note('error_code: ' + step.error_code + ' · retryable: ' + text(step.retryable), 'error'));
      if (step.action == null) body.append(note('未记录可解析的 action。', 'warn'));
      else {
        body.append(el('div', 'json-label', 'Arguments'), json(action.args));
        if (action.thought != null) body.append(details('Action thought · 记录中的简述', el('div', 'text-block', action.thought)));
      }
      if (validIndex(responseIndex, history)) {
        body.append(details('模型原始响应 · message #' + responseIndex, message(history[responseIndex], responseIndex)));
      } else body.append(note('模型响应索引缺失或越界：' + text(responseIndex), 'warn'));
      const observation = el('div', 'observation'); observation.append(el('h3', '', 'Observation'));
      const observationIndex = step.observation_message_index;
      if (validIndex(observationIndex, history)) observation.append(message(history[observationIndex], observationIndex));
      else if (observationIndex == null) observation.append(el('p', 'muted', '该 step 没有记录 observation（例如 runtime finish）。'));
      else observation.append(note('Observation 索引缺失或越界：' + text(observationIndex), 'warn'));
      body.append(observation); card.append(body); parent.append(card);
    });
  }

  function renderResult(record, trajectory) {
    const parent = $('result-content'), panel = el('div', 'panel');
    panel.append(status(trajectory), note('finish 只代表流程正常终止，不等于任务验证通过或 Judge 高分。'));
    if (trajectory.final_answer != null) panel.append(el('div', 'outcome', trajectory.final_answer));
    else panel.append(el('p', 'empty', '未记录 final_answer。'));
    parent.append(panel);
    const lastVisual = [...list(trajectory.messages)].reverse().find(m =>
      obj(m).role === 'observation' && list(obj(m).content).some(b => obj(b).type === 'image'));
    if (lastVisual) {
      const visual = el('div', 'panel last-images');
      visual.append(el('h3', '', '最后一次图片 observation · ' + (lastVisual.name || '未命名')),
                    el('p', 'muted', '这是记录中的最后图片，不自动认定为最终交付文件。'));
      content(visual, lastVisual.content.filter(b => obj(b).type === 'image')); parent.append(visual);
    }
    parent.append(details('Trajectory metadata · 控制信息与时间', json({
      episode_id: trajectory.episode_id, started_at: trajectory.started_at, completed_at: trajectory.completed_at,
      termination_reason: trajectory.termination_reason, success: trajectory.success, metadata: trajectory.metadata,
    })));
    const summary = obj(record.summary);
    if (record.summary) {
      parent.append(details('运行摘要 / 模型调用用量（补充记录）', body => {
        body.append(note('摘要是补充数据。若含多个轨迹版本，不自动将其字段归属于所选版本。'));
        body.append(json(summary));
      }));
    }
  }

  function renderJudge(record) {
    const parent = $('judge-content'), judge = obj(record.judge), panel = el('div', 'panel');
    if (!judge.present) panel.append(el('p', 'empty', '未记录 Judge 评分。本报告不会自动评分，也不会把缺失值当作 0。'));
    else {
      const head = el('div', 'score-header');
      head.append(el('span', 'score-number', fmt(judge.overall)), el('div', 'score-caption', '行级 overall · ' + judge.score_key + '\n原样展示；缺失值为 —'));
      panel.append(head);
    }
    panel.append(note(judge.scope_note, record.variants.length > 1 ? 'warn' : ''));
    if (record.refined != null) panel.append(el('p', '', '_refined: ' + text(record.refined)));
    if (record.refine_note) panel.append(details('Refine 记录 / 诊断', el('div', 'text-block', record.refine_note)));
    const rubric = obj(judge.rubric), criteria = list(rubric.criteria), scores = obj(judge.scores);
    const modelScores = obj(judge.model_scores), normalized = obj(judge.normalized_scores);
    const ids = [...new Set([...criteria.map(c => obj(c).id).filter(x => typeof x === 'string'), ...Object.keys(scores), ...Object.keys(modelScores), ...Object.keys(normalized)])];
    if (rubric.score_range) panel.append(el('p', 'muted', 'Rubric 原始尺度：' + text(rubric.score_range.min) + ' — ' + text(rubric.score_range.max)));
    if (ids.length) {
      const wrapper = el('div', 'table-wrap'), table = el('table'), thead = el('thead'), row = el('tr');
      ['Criterion / 标准', '原始分', '模型分', '归一化分'].forEach(label => row.append(el('th', '', label)));
      thead.append(row); table.append(thead); const tbody = el('tbody');
      for (const id of ids) {
        const tr = el('tr'), criterion = el('td', 'criterion', id);
        const description = criteria.find(c => obj(c).id === id)?.description;
        if (description) criterion.append(el('span', 'criterion-description', description));
        tr.append(criterion, el('td', 'number', fmt(scores[id])), el('td', 'number', fmt(modelScores[id])));
        const cell = el('td', 'number', fmt(normalized[id]));
        if (finite(normalized[id]) && normalized[id] >= 0 && normalized[id] <= 1) {
          const meter = el('meter'); meter.min = 0; meter.max = 1; meter.value = normalized[id];
          meter.setAttribute('aria-label', id + ' 归一化分'); cell.append(meter);
        }
        tr.append(cell); tbody.append(tr);
      }
      table.append(tbody); wrapper.append(table); panel.append(wrapper);
    }
    if (judge.rationale != null) {
      panel.append(el('h3', '', 'Judge rationale'), el('div', 'text-block', judge.rationale));
    }
    if (judge.rubric) panel.append(details('原始 Judge rubric', json(judge.rubric)));
    parent.append(panel);
    const replay = el('div', 'panel'); replay.append(el('h3', '', 'Replay verification · 独立验证'));
    if (record.replay == null) replay.append(el('p', 'muted', '未记录 replay_verification。'));
    else { replay.append(badge(obj(record.replay).status || '已记录'), json(record.replay)); }
    parent.append(replay);
    if (Object.keys(obj(record.metadata)).length) parent.append(details('Pipeline 行字段（不含轨迹正文）', json(record.metadata)));
  }

  function render() {
    if ($('detail-dialog').open) $('detail-dialog').close();
    recordList();
    const record = records[selectedRecord], variant = record.variants[selectedVariant];
    const trajectory = obj(variant?.trajectory);
    $('task-title').textContent = text(record.task_id);
    const meta = $('record-meta'); meta.replaceChildren();
    const steps = list(trajectory.steps);
    const images = list(trajectory.messages).flatMap(m => list(obj(m).content)).filter(b => obj(b).type === 'image').length;
    meta.append(el('span', '', record.env_id), el('span', '', trajectory.episode_id || '无 episode'), status(trajectory),
                el('span', '', steps.length + ' steps · ' + images + ' images · ' + steps.filter(s => obj(s).tool_ok === false).length + ' errors'),
                el('span', '', elapsed(trajectory)));
    const summary = obj(record.summary);
    if (summary.model) meta.append(el('span', '', summary.model));
    if (finite(summary.total_tokens)) meta.append(el('span', '', fmt(summary.total_tokens) + ' tokens · 摘要'));
    $('variant-tabs').replaceChildren();
    if (record.variants.length > 1) record.variants.forEach((item, index) => {
      const button = el('button', '', item.label + ' · ' + list(item.trajectory.steps).length + ' steps');
      button.type = 'button'; button.setAttribute('aria-pressed', String(index === selectedVariant));
      button.addEventListener('click', () => { selectedVariant = index; render(); }); $('variant-tabs').append(button);
    });
    ['warnings', 'task-content', 'timeline', 'result-content', 'judge-content', 'messages-content'].forEach(id => $(id).replaceChildren());
    for (const warning of record.warnings) $('warnings').append(note(warning, 'warn'));
    if (!variant) $('warnings').append(note('这一行没有可展示的 trajectory；保留行级评分及元数据。', 'warn'));
    renderTask(record, trajectory); renderTimeline(trajectory); renderResult(record, trajectory); renderJudge(record);
    $('messages-content').append(details('完整消息序列 · ' + list(trajectory.messages).length + ' 条（含未绑定 step 的消息）', body => messages(body, list(trajectory.messages))));
  }
  render();
})();
