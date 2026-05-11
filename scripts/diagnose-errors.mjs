#!/usr/bin/env node
// Sequential FP/FN dump for the fraud-score endpoint.
//
// Reads test/test-data.json, hits POST http://localhost:9999/fraud-score for
// every entry, and writes benchmarks/errors-dump.json with each disagreement
// (FP, FN, or HTTP error). Use this to isolate detection failures without
// the contention of the k6 load run.
//
// Usage:
//   node scripts/diagnose-errors.mjs [--limit N] [--url URL]

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');

const argv = new Map();
for (let i = 2; i < process.argv.length; i += 2) {
    argv.set(process.argv[i].replace(/^--/, ''), process.argv[i + 1]);
}
const URL = argv.get('url') || 'http://localhost:9999/fraud-score';
const LIMIT = Number(argv.get('limit') || '0') || Infinity;
const OUT = path.join(ROOT, 'benchmarks', 'errors-dump.json');

const dataPath = path.join(ROOT, 'test', 'test-data.json');
const dataset = JSON.parse(fs.readFileSync(dataPath, 'utf8'));
const entries = dataset.entries;
console.error(`Loaded ${entries.length} entries from ${dataPath}`);

const errors = [];
let tp = 0, tn = 0, fp = 0, fn = 0, errs = 0;
let i = 0;

const stop = Math.min(entries.length, LIMIT);
const t0 = Date.now();

for (; i < stop; i++) {
    const entry = entries[i];
    const body = JSON.stringify(entry.request);
    let res, json;
    try {
        res = await fetch(URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body,
        });
        if (!res.ok) throw new Error(`status ${res.status}`);
        json = await res.json();
    } catch (e) {
        errs++;
        errors.push({
            type: 'ERR',
            idx: i,
            request: entry.request,
            expected_approved: entry.expected_approved,
            expected_fraud_score: entry.expected_fraud_score,
            error: String(e),
        });
        continue;
    }

    const expected = entry.expected_approved;
    const got = json.approved;
    if (expected === got) {
        if (got) tn++; else tp++;
    } else if (got) {
        // got approved but expected denied → fraud went through (FN)
        fn++;
        errors.push({
            type: 'FN',
            idx: i,
            request: entry.request,
            expected_approved: expected,
            expected_fraud_score: entry.expected_fraud_score,
            got_approved: got,
            got_fraud_score: json.fraud_score,
        });
    } else {
        // got denied but expected approved → false block (FP)
        fp++;
        errors.push({
            type: 'FP',
            idx: i,
            request: entry.request,
            expected_approved: expected,
            expected_fraud_score: entry.expected_fraud_score,
            got_approved: got,
            got_fraud_score: json.fraud_score,
        });
    }

    if ((i + 1) % 5000 === 0) {
        const dt = ((Date.now() - t0) / 1000).toFixed(1);
        console.error(`  [${i + 1}/${stop}] tp=${tp} tn=${tn} fp=${fp} fn=${fn} errs=${errs} (${dt}s)`);
    }
}

const dt = ((Date.now() - t0) / 1000).toFixed(1);
const N = tp + tn + fp + fn + errs;
const E = fp * 1 + fn * 3 + errs * 5;
const failureRate = N > 0 ? (fp + fn + errs) / N : 0;

const summary = {
    url: URL,
    total: N,
    duration_seconds: Number(dt),
    breakdown: { tp, tn, fp, fn, errs },
    weighted_E: E,
    failure_rate: +(failureRate * 100).toFixed(2),
    errors,
};

fs.mkdirSync(path.dirname(OUT), { recursive: true });
fs.writeFileSync(OUT, JSON.stringify(summary, null, 2));
console.error(`\nWrote ${errors.length} error records → ${OUT}`);
console.error(`tp=${tp} tn=${tn} fp=${fp} fn=${fn} errs=${errs} | E=${E} | failure_rate=${(failureRate * 100).toFixed(2)}%`);
