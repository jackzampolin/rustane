// rustane bench — Cloudflare Worker API
// Handles benchmark submission, validation, and retrieval.

// Minimum ms/step per benchmark (no chip can beat ANE dispatch overhead)
const MIN_MS = {
  sweep_600m_a: 100, sweep_600m_b: 100, sweep_600m_c: 100, sweep_600m_d: 100, sweep_600m_e: 100,
  sweep_1b_a: 300, sweep_1b_b: 300, sweep_1b_c: 300, sweep_1b_d: 300, sweep_1b_e: 300,
  sweep_1_5b_a: 500, sweep_1_5b_b: 500, sweep_1_5b_c: 500, sweep_1_5b_d: 500, sweep_1_5b_e: 500,
  sweep_3b_a: 1000, sweep_3b_b: 1000, sweep_3b_c: 1000, sweep_3b_d: 1000, sweep_3b_e: 1000,
  sweep_5b_a: 2000, sweep_5b_b: 2000, sweep_5b_c: 2000, sweep_5b_d: 2000, sweep_5b_e: 2000,
  sweep_7b: 5000, sweep_10b: 8000,
  fwd_5b: 500, fwd_7b: 1000, fwd_10b: 1500,
  fwd_13b: 5000, fwd_15b: 7000, fwd_20b: 10000, fwd_25b: 15000, fwd_30b: 20000,
};

// Minimum RAM (GB) per benchmark
const MIN_RAM = {
  sweep_600m_a: 12, sweep_600m_b: 12, sweep_600m_c: 12, sweep_600m_d: 12, sweep_600m_e: 12,
  sweep_1b_a: 25, sweep_1b_b: 25, sweep_1b_c: 25, sweep_1b_d: 25, sweep_1b_e: 25,
  sweep_1_5b_a: 30, sweep_1_5b_b: 30, sweep_1_5b_c: 30, sweep_1_5b_d: 30, sweep_1_5b_e: 30,
  sweep_3b_a: 55, sweep_3b_b: 55, sweep_3b_c: 55, sweep_3b_d: 55, sweep_3b_e: 55,
  sweep_5b_a: 85, sweep_5b_b: 85, sweep_5b_c: 85, sweep_5b_d: 85, sweep_5b_e: 85,
  fwd_7b: 31, fwd_10b: 46, fwd_13b: 60, fwd_15b: 70, fwd_20b: 93, fwd_25b: 110, fwd_30b: 120,
};

function jsonResponse(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function checkPlausibility(body) {
  const bench = body.benchmark;

  // Check minimum timing
  if (MIN_MS[bench] && body.results.ms_per_step < MIN_MS[bench]) {
    return { reject: true, reason: `ms_per_step too fast for ${bench}` };
  }

  // Check RAM vs reported hardware
  if (MIN_RAM[bench] && body.hardware.ram_gb < MIN_RAM[bench]) {
    return { reject: true, reason: `${bench} needs ${MIN_RAM[bench]}GB, reported ${body.hardware.ram_gb}GB` };
  }

  // Component times must roughly sum to total (training benchmarks only)
  const sum = body.results.ms_fwd + body.results.ms_bwd + body.results.ms_upd;
  if (sum > 0 && body.results.ms_bwd > 0) {
    if (Math.abs(sum - body.results.ms_per_step) / body.results.ms_per_step > 0.25) {
      return { reject: true, reason: "Component times don't sum to ms_per_step" };
    }
  }

  // Loss trace sanity: all values must be finite
  const allFinite = body.loss_trace.every((l) => isFinite(l));
  if (!allFinite) {
    return { reject: true, reason: "Loss trace contains NaN/Inf" };
  }

  return { reject: false, verified: allFinite };
}

async function updateIndex(env, benchmark, entry) {
  const key = `index:${benchmark}`;
  const index = (await env.LEADERBOARD.get(key, "json")) || [];
  index.push(entry);

  // Sort: training by tok_per_s descending, forward by ms ascending
  if (benchmark.startsWith("fwd_")) {
    index.sort((a, b) => a.ms_per_step - b.ms_per_step);
  } else {
    index.sort((a, b) => b.tok_per_s - a.tok_per_s);
  }

  // Keep top 100 per benchmark
  await env.LEADERBOARD.put(key, JSON.stringify(index.slice(0, 100)));
}

async function handleSubmit(request, env) {
  let body;
  try {
    body = await request.json();
  } catch {
    return jsonResponse({ error: "Invalid JSON" }, 400);
  }

  // Schema validation
  if (
    !body.schema_version ||
    !body.benchmark ||
    !body.config ||
    !body.results ||
    !body.hardware ||
    !body.loss_trace
  ) {
    return jsonResponse({ error: "Missing required fields" }, 400);
  }

  // tok_per_s consistency check
  if (body.results.ms_per_step > 0) {
    const expected = (body.config.seq * 1000) / body.results.ms_per_step;
    if (Math.abs(expected - body.results.tok_per_s) / expected > 0.05) {
      return jsonResponse(
        { error: "tok_per_s inconsistent with ms_per_step" },
        400
      );
    }
  }

  // Loss must decrease for training benchmarks
  if (body.benchmark.startsWith("sweep_") && body.results.loss_delta >= 0) {
    return jsonResponse({ error: "Loss did not decrease" }, 400);
  }

  // Plausibility checks
  const plausibility = checkPlausibility(body);
  if (plausibility.reject) {
    return jsonResponse({ error: plausibility.reason }, 400);
  }

  // Rate limit: 1 per hardware + benchmark per hour
  const rateKey = `rate:${body.hardware.chip}:${body.hardware.ram_gb}:${body.benchmark}`;
  const existing = await env.LEADERBOARD.get(rateKey);
  if (existing) {
    return jsonResponse(
      { error: "Rate limited. Try again in 1 hour." },
      429
    );
  }
  await env.LEADERBOARD.put(rateKey, "1", { expirationTtl: 3600 });

  // Generate ID and store
  const id = crypto.randomUUID().slice(0, 8);
  const entry = {
    id,
    ...body,
    submitted_at: new Date().toISOString(),
    verified: plausibility.verified,
  };

  await env.LEADERBOARD.put(`result:${id}`, JSON.stringify(entry));

  // Update per-benchmark index
  await updateIndex(env, body.benchmark, {
    id,
    name: body.submitter?.name || "Anonymous",
    x_handle: body.submitter?.x_handle || "",
    chip: body.hardware.chip,
    ram_gb: body.hardware.ram_gb,
    benchmark: body.benchmark,
    ms_per_step: body.results.ms_per_step,
    tok_per_s: body.results.tok_per_s,
    params_m: body.config.params_m,
    loss_delta: body.results.loss_delta,
    git_sha: body.git_sha,
    verified: plausibility.verified,
    submitted_at: entry.submitted_at,
  });

  // Update master index (list of all benchmarks)
  const masterKey = "index:__all__";
  const master = (await env.LEADERBOARD.get(masterKey, "json")) || [];
  if (!master.includes(body.benchmark)) {
    master.push(body.benchmark);
    master.sort();
    await env.LEADERBOARD.put(masterKey, JSON.stringify(master));
  }

  return jsonResponse({
    id,
    status: "accepted",
    verified: plausibility.verified,
    url: `https://bench.rustane.org/?id=${id}`,
  });
}

async function handleResults(request, env) {
  const url = new URL(request.url);
  const benchmark = url.searchParams.get("benchmark");

  if (benchmark) {
    const index = await env.LEADERBOARD.get(`index:${benchmark}`, "json");
    return jsonResponse(index || []);
  }

  // Return all benchmarks
  const master =
    (await env.LEADERBOARD.get("index:__all__", "json")) || [];
  const results = {};
  for (const bench of master) {
    results[bench] =
      (await env.LEADERBOARD.get(`index:${bench}`, "json")) || [];
  }
  return jsonResponse(results);
}

async function handleResultDetail(request, env, id) {
  const result = await env.LEADERBOARD.get(`result:${id}`, "json");
  if (!result) return jsonResponse({ error: "Not found" }, 404);
  return jsonResponse(result);
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const path = url.pathname;

    // CORS
    const corsHeaders = {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
    };

    if (request.method === "OPTIONS") {
      return new Response(null, { headers: corsHeaders });
    }

    let response;
    try {
      if (request.method === "POST" && path === "/api/submit") {
        response = await handleSubmit(request, env);
      } else if (request.method === "GET" && path === "/api/results") {
        response = await handleResults(request, env);
      } else if (
        request.method === "GET" &&
        path.startsWith("/api/result/")
      ) {
        const id = path.split("/").pop();
        response = await handleResultDetail(request, env, id);
      } else {
        response = jsonResponse({ error: "Not found" }, 404);
      }
    } catch (err) {
      response = jsonResponse({ error: "Internal error" }, 500);
    }

    // Add CORS headers
    Object.entries(corsHeaders).forEach(([k, v]) =>
      response.headers.set(k, v)
    );
    return response;
  },
};
