const express  = require('express');
const cors     = require('cors');
const crypto   = require('crypto');
const fs       = require('fs');
const path     = require('path');
const multer   = require('multer');
require('dotenv').config();

const { GoogleGenerativeAI } = require('@google/generative-ai');
const OpenAI = require('openai');

let officeParser;
try {
  officeParser = require('officeparser');
} catch (e) {
  console.warn("⚠️ 'officeparser' eklentisi eksik. Word/PPT belgelerini okumak için: npm install officeparser");
}

const app = express();
app.use(cors());
app.use(express.json({ limit: '20mb' }));
app.use(express.static(path.join(__dirname, 'public')));

// ================= CONFIG =================
const GEMINI_KEY = process.env.GEMINI_API_KEY;
const OPENAI_KEY = process.env.OPENAI_API_KEY;
const OLLAMA_URL = process.env.OLLAMA_URL || 'http://localhost:11434/api/chat';
const PORT       = process.env.PORT || 3000;
const DATA_FILE  = path.join(__dirname, 'userdata.json');
const UPLOAD_DIR = path.join(__dirname, 'uploads');

if (!fs.existsSync(UPLOAD_DIR)) fs.mkdirSync(UPLOAD_DIR);

const genAI  = GEMINI_KEY ? new GoogleGenerativeAI(GEMINI_KEY) : null;
const openai = OPENAI_KEY ? new OpenAI({ apiKey: OPENAI_KEY }) : null;

// ================= LRU CACHE =================
const LRU_MAX  = 200;
const lruCache = new Map();

function lruGet(key) {
  const entry = lruCache.get(key);
  if (!entry) return null;
  lruCache.delete(key);
  lruCache.set(key, entry);
  return entry.value;
}

function lruSet(key, value) {
  if (lruCache.has(key)) lruCache.delete(key);
  else if (lruCache.size >= LRU_MAX) {
    lruCache.delete(lruCache.keys().next().value);
  }
  lruCache.set(key, { value, ts: Date.now() });
}

function hashMsg(messages, activeModels) {
  const str = messages.map(m => m.role + ':' + (typeof m.content === 'string' ? m.content : JSON.stringify(m.content))).join('|') + JSON.stringify(activeModels);
  return crypto.createHash('sha256').update(str).digest('hex');
}

// ================= PERSISTENT USER PREFS =================
let userPrefs = {};

function loadData() {
  try {
    if (fs.existsSync(DATA_FILE)) {
      userPrefs = JSON.parse(fs.readFileSync(DATA_FILE, 'utf8'));
      console.log(`📂 Kullanıcı verileri yüklendi: ${Object.keys(userPrefs).length} kullanıcı`);
    }
  } catch (e) {
    console.warn('userdata.json okunamadı, boş başlıyor:', e.message);
    userPrefs = {};
  }
}

function saveData() {
  try {
    fs.writeFileSync(DATA_FILE, JSON.stringify(userPrefs, null, 2));
  } catch (e) {
    console.error('userdata.json yazılamadı:', e.message);
  }
}

loadData();

function getUserPrefs(userId) {
  if (!userPrefs[userId]) {
    userPrefs[userId] = {
      votes:      { gemini: 0, gpt: 0, ollama: 0 },
      totalChats: 0,
      lastSeen:   null,
    };
  }
  return userPrefs[userId];
}

function recordModelWin(userId, model) {
  const u = getUserPrefs(userId);
  u.votes[model] = (u.votes[model] || 0) + 1;
  u.totalChats++;
  u.lastSeen = new Date().toISOString();
  saveData();
}

// ================= QUALITY SCORE =================
function qualityScore(text) {
  if (!text || !text.trim()) return 0;
  const lenScore = Math.min(Math.log(text.length + 1) * 50, 400);
  const words   = text.toLowerCase().match(/\b\w+\b/g) || [];
  const unique  = new Set(words).size;
  const divScore = words.length > 0 ? (unique / words.length) * 100 : 0;
  const codeBonus = (text.match(/```/g) || []).length * 15;
  const errorPenalty = text.length < 20 || /^\s*(error|hata|undefined)\s*$/i.test(text) ? -200 : 0;
  return lenScore + divScore + codeBonus + errorPenalty;
}

// ================= MODEL CALLERS =================

// Gemini model fallback sırası: en iyi → en düşük kota tüketen
const GEMINI_MODELS = [
  'gemini-2.5-flash',
  'gemini-2.0-flash',
  'gemini-1.5-flash',
  'gemini-1.5-flash-8b',
];
// Her model için devre dışı kalma süresi (429 alınca)
const geminiDisabledUntil = {};

function getAvailableGeminiModel() {
  const now = Date.now();
  for (const m of GEMINI_MODELS) {
    if (!geminiDisabledUntil[m] || now > geminiDisabledUntil[m]) return m;
  }
  // Hepsi disabled — en erken açılacak olanı döndür (ve bekleme süresini logla)
  const soonest = GEMINI_MODELS.reduce((a, b) =>
    (geminiDisabledUntil[a] || 0) < (geminiDisabledUntil[b] || 0) ? a : b
  );
  const waitSec = Math.ceil((geminiDisabledUntil[soonest] - now) / 1000);
  throw new Error(`Tüm Gemini modelleri kota aşımında. En erken ${waitSec}s sonra (${soonest}) kullanılabilir.`);
}

function disableGeminiModel(modelName, retryAfterMs) {
  geminiDisabledUntil[modelName] = Date.now() + (retryAfterMs || 60 * 60 * 1000);
  console.warn(`⚠️  Gemini ${modelName} devre dışı, ${Math.round((retryAfterMs||3600000)/60000)} dk sonra tekrar denenecek`);
}

// 429 yanıtından retry-delay süresini parse et (saniye cinsinden)
function parseRetryDelay(err) {
  try {
    const msg = err.message || '';
    const m = msg.match(/retry[^:]*:\s*["']?(\d+)s/i) || msg.match(/"retryDelay":"(\d+)s"/);
    if (m) return parseInt(m[1]) * 1000 + 5000; // +5s buffer
  } catch {}
  return 65 * 1000; // varsayılan 65 saniye
}

async function askGemini(messages, systemPrompt) {
  if (!genAI) throw new Error('GEMINI_API_KEY eksik');

  const modelName = getAvailableGeminiModel();
  const model = genAI.getGenerativeModel({
    model: modelName,
    systemInstruction: systemPrompt || 'Sen yardımcı bir asistansın.',
    generationConfig: { temperature: 0.7, maxOutputTokens: 1024 },
  });

  const history = messages.slice(0, -1).map(m => ({
    role:  m.role === 'assistant' ? 'model' : 'user',
    parts: [{ text: typeof m.content === 'string' ? m.content : JSON.stringify(m.content) }],
  }));

  const lastMsg = messages[messages.length - 1].content;
  const chat    = model.startChat({ history });

  try {
    const result = await chat.sendMessage(typeof lastMsg === 'string' ? lastMsg : JSON.stringify(lastMsg));
    console.log(`✅ Gemini model kullanıldı: ${modelName}`);
    return result.response.text();
  } catch (err) {
    const is429 = err.message?.includes('429') || err.status === 429;
    if (is429) {
      disableGeminiModel(modelName, parseRetryDelay(err));
      // Bir sonraki modelle tekrar dene
      return askGemini(messages, systemPrompt);
    }
    throw err;
  }
}

// Gemini Vision — image + text (aynı fallback mantığı)
async function askGeminiVision(imageBase64, mimeType, userText, systemPrompt) {
  if (!genAI) throw new Error('GEMINI_API_KEY eksik');

  const modelName = getAvailableGeminiModel();
  const model = genAI.getGenerativeModel({ model: modelName });
  const prompt = systemPrompt
    ? `${systemPrompt}\n\n${userText || 'Bu resmi analiz et.'}`
    : (userText || 'Bu resmi analiz et.');

  try {
    const result = await model.generateContent([
      { inlineData: { data: imageBase64, mimeType } },
      prompt,
    ]);
    return result.response.text();
  } catch (err) {
    const is429 = err.message?.includes('429') || err.status === 429;
    if (is429) {
      disableGeminiModel(modelName, parseRetryDelay(err));
      return askGeminiVision(imageBase64, mimeType, userText, systemPrompt);
    }
    throw err;
  }
}

let gptDisabledUntil = 0;
async function askGPT(messages, systemPrompt) {
  if (!openai) throw new Error('OPENAI_API_KEY eksik');
  if (Date.now() < gptDisabledUntil) {
    throw new Error(`GPT kota aşıldı, ${Math.ceil((gptDisabledUntil - Date.now())/60000)} dk sonra tekrar denenecek`);
  }

  const msgs = [
    { role: 'system', content: systemPrompt || 'Sen yardımcı bir asistansın.' },
    ...messages.map(m => ({ role: m.role === 'assistant' ? 'assistant' : 'user', content: typeof m.content === 'string' ? m.content : JSON.stringify(m.content) })),
  ];

  try {
    const res = await openai.chat.completions.create({
      model:       'gpt-4o-mini',
      messages:    msgs,
      max_tokens:  1024,
      temperature: 0.7,
    });
    return res.choices[0].message.content;
  } catch (err) {
    if (err.status === 429 || err.code === 'insufficient_quota') {
      gptDisabledUntil = Date.now() + 30 * 60 * 1000;
      console.warn('⚠️  GPT kota aşıldı, 30 dakika devre dışı.');
    }
    throw err;
  }
}

let ollamaAvailable = true;
let ollamaCheckAt   = 0;
async function askOllama(messages, systemPrompt) {
  if (!ollamaAvailable && Date.now() < ollamaCheckAt) {
    throw new Error('Ollama erişilemiyor (skip)');
  }

  const body = {
    model:  process.env.OLLAMA_MODEL || 'llama3',
    stream: false,
    messages: [
      { role: 'system', content: systemPrompt || 'Sen yardımcı bir asistansın.' },
      ...messages.map(m => ({ role: m.role === 'assistant' ? 'assistant' : 'user', content: typeof m.content === 'string' ? m.content : JSON.stringify(m.content) })),
    ],
  };

  try {
    const r = await fetch(OLLAMA_URL, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(body),
      signal:  AbortSignal.timeout(60000),
    });

    if (!r.ok) throw new Error(`Ollama HTTP ${r.status}`);
    const d = await r.json();
    ollamaAvailable = true;
    return d.message?.content || d.response;
  } catch (err) {
    if (err.name === 'TimeoutError' || err.message.includes('aborted')) {
      ollamaAvailable = false;
      ollamaCheckAt   = Date.now() + 5 * 60 * 1000;
      console.warn('⚠️  Ollama timeout — 5 dk sonra tekrar denenecek');
    }
    throw err;
  }
}

// ================= IMAGE UPLOAD (Vision) =================

const IMAGE_MIMES = ['image/jpeg', 'image/png', 'image/webp', 'image/gif'];

const imageUpload = multer({
  dest: UPLOAD_DIR,
  limits: { fileSize: 10 * 1024 * 1024 },
  fileFilter: (req, file, cb) => cb(null, IMAGE_MIMES.includes(file.mimetype)),
});

app.post('/api/image', imageUpload.single('image'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: 'Geçersiz dosya. Sadece JPG, PNG, WEBP, GIF desteklenir.' });
    if (!genAI)    return res.status(500).json({ error: 'Görüntü analizi için GEMINI_API_KEY gerekli.' });

    const imageData = fs.readFileSync(req.file.path);
    const base64    = imageData.toString('base64');
    const mimeType  = req.file.mimetype;
    const userText  = req.body.text || '';
    const systemPrompt = req.body.systemPrompt || '';

    if (fs.existsSync(req.file.path)) fs.unlinkSync(req.file.path);

    const t0   = Date.now();
    const text = await askGeminiVision(base64, mimeType, userText, systemPrompt);
    res.json({ text, model: 'gemini', latency: Date.now() - t0 });
  } catch (err) {
    console.error('Image hatası:', err.message);
    if (req.file && fs.existsSync(req.file.path)) fs.unlinkSync(req.file.path);
    res.status(500).json({ error: err.message });
  }
});

// ================= DOCUMENT UPLOAD (PDF, DOCX, PPTX, TXT) =================

const allowedMimes = [
  'application/pdf',
  'text/plain',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document',   // .docx
  'application/vnd.openxmlformats-officedocument.presentationml.presentation', // .pptx
];

// FIX: multer'a orijinal uzantıyı koruyan özel storage tanımlıyoruz.
// Uzantısız geçici dosya adı officeparser'ın hata vermesine yol açıyordu.
const docStorage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, UPLOAD_DIR),
  filename: (req, file, cb) => {
    const ext  = path.extname(file.originalname).toLowerCase(); // .docx / .pptx vb.
    const name = crypto.randomBytes(16).toString('hex') + ext;
    cb(null, name);
  },
});

const upload = multer({
  storage: docStorage,
  limits:  { fileSize: 10 * 1024 * 1024 },
  fileFilter: (req, file, cb) => cb(null, allowedMimes.includes(file.mimetype)),
});

function extractPDFText(filePath) {
  const buf = fs.readFileSync(filePath);
  const str = buf.toString('latin1');
  const chunks = [];
  const btEt = /BT([\s\S]*?)ET/g;
  let m;
  while ((m = btEt.exec(str)) !== null) {
    const block = m[1];
    const tj = /\(([^)]*)\)\s*(?:Tj|')|(\[([^\]]*)\])\s*TJ/g;
    let t;
    while ((t = tj.exec(block)) !== null) {
      if (t[1]) chunks.push(t[1].replace(/\\n/g,' ').replace(/\\r/g,' '));
      else if (t[3]) {
        (t[3].match(/\(([^)]*)\)/g) || []).forEach(p => chunks.push(p.slice(1,-1)));
      }
    }
  }
  const text = chunks.join(' ').replace(/\s+/g,' ').trim();
  return text.length > 30 ? text : '[PDF metni çıkarılamadı — taranmış/görüntü tabanlı olabilir.]';
}

app.post('/api/upload', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: 'Geçersiz dosya. Lütfen sadece PDF, TXT, DOCX veya PPTX yükleyin.' });

    let text = '';
    const mime = req.file.mimetype;

    if (mime === 'text/plain') {
      text = fs.readFileSync(req.file.path, 'utf8');
    } else if (mime === 'application/pdf') {
      text = extractPDFText(req.file.path);
    } else {
      // DOCX / PPTX — officeparser artık uzantıyı görebilir
      if (officeParser) {
        text = await new Promise((resolve, reject) => {
          officeParser.parseOffice(req.file.path, function(data, err) {
            if (err) return reject(new Error('Belge okunamadı: ' + err));
            resolve(data);
          });
        });
      } else {
        text = '[Sistem Notu: Dosya alındı ancak officeparser kurulu değil. `npm install officeparser` çalıştırın.]';
      }
    }

    if (fs.existsSync(req.file.path)) fs.unlinkSync(req.file.path);
    res.json({ text, filename: req.file.originalname });
  } catch (err) {
    console.error('Upload hatası:', err.message);
    if (req.file && fs.existsSync(req.file.path)) fs.unlinkSync(req.file.path);
    res.status(500).json({ error: err.message });
  }
});

// ================= CHAT ROUTER =================
app.post('/api/chat', async (req, res) => {
  const { messages = [], systemPrompt, userId = 'anon', activeModels = {} } = req.body;

  if (!messages.length) return res.status(400).json({ error: 'Mesaj listesi boş.' });

  const cacheKey = hashMsg(messages, activeModels);
  const cached   = lruGet(cacheKey);
  if (cached) {
    console.log(`⚡ Cache hit (${cacheKey.slice(0,8)})`);
    return res.json({ ...cached, cached: true });
  }

  const allTasks = [
    { id: 'gemini', name: 'gemini', fn: () => askGemini(messages, systemPrompt) },
    { id: 'gpt',    name: 'gpt',    fn: () => askGPT(messages, systemPrompt)    },
    { id: 'ollama', name: 'ollama', fn: () => askOllama(messages, systemPrompt) },
  ];

  const tasksToRun = allTasks.filter(t => activeModels[t.id]);

  if (tasksToRun.length === 0) {
    return res.status(400).json({ error: 'Lütfen en az bir model seçin.' });
  }

  const startAll = Date.now();
  const tasks = tasksToRun.map(({ name, fn }) => {
    const t0 = Date.now();
    return fn()
      .then(text => ({ model: name, text, latency: Date.now() - t0 }))
      .catch(err => {
        const reason = err.message?.slice(0, 120) || 'bilinmeyen hata';
        console.warn(`⚠️  ${name} hata: ${reason}`);
        return { __error: true, model: name, reason };
      });
  });

  const rawResults = await Promise.all(tasks);
  const responses  = rawResults.filter(r => r && !r.__error);
  const failures   = rawResults.filter(r => r && r.__error);

  if (!responses.length) {
    const errDetails = failures.map(f => `${f.model}: ${f.reason}`).join(' | ');
    const msg = errDetails
      ? `Tüm modeller hata verdi → ${errDetails}`
      : 'Seçili modeller yanıt veremedi. API anahtarlarını ve model seçimini kontrol et.';
    return res.status(500).json({ error: msg });
  }

  const prefs = getUserPrefs(userId).votes;
  const scored = responses.map(r => {
    const q     = qualityScore(r.text);
    const pref  = (prefs[r.model] || 0) * 8;
    const speed = Math.max(0, 300 - r.latency) / 10;
    return { ...r, totalScore: q + pref + speed };
  });

  scored.sort((a, b) => b.totalScore - a.totalScore);
  const best = scored[0];

  const allResults = scored.map(r => ({
    model:   r.model,
    score:   Math.round(r.totalScore),
    latency: r.latency,
  }));

  console.log(`✅ [${userId}] Winner: ${best.model} (score:${Math.round(best.totalScore)}, ${best.latency}ms) | Total: ${Date.now()-startAll}ms`);

  recordModelWin(userId, best.model);

  const response = {
    text:       best.text,
    model:      best.model,
    latency:    best.latency,
    allResults,
  };

  lruSet(cacheKey, response);
  res.json(response);
});

// ================= STATS =================
app.get('/api/stats', (req, res) => {
  const { userId = 'anon' } = req.query;
  const u = getUserPrefs(userId);
  res.json({
    userId,
    votes:      u.votes,
    totalChats: u.totalChats,
    cacheSize:  lruCache.size,
    uptime:     Math.round(process.uptime()) + 's',
  });
});

// ================= START =================
app.listen(PORT, () => {
  console.log(`🚀 Server: http://localhost:${PORT}`);
  console.log(`🤖 Gemini 2.5 Flash: ${GEMINI_KEY ? '✅' : '❌ (GEMINI_API_KEY eksik)'}`);
  console.log(`🤖 GPT-4o-mini:      ${OPENAI_KEY ? '✅' : '❌ (OPENAI_API_KEY eksik)'}`);
  console.log(`🤖 Ollama (llama3):  kontrol ediliyor...`);
  fetch(OLLAMA_URL.replace('/api/chat', '/api/tags'))
    .then(r => r.json())
    .then(d => console.log(`   Ollama modelleri: ${d.models?.map(m=>m.name).join(', ') || 'bulunamadı'}`))
    .catch(() => console.log(`   Ollama: bağlantı kurulamadı (${OLLAMA_URL})`));
});
