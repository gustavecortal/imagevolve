// The search: random recipes, or children of a kept entry, and a pool kept varied by comparing the candidates'
// image embeddings. An entry is { recipe, emb, score }: emb the candidate's unit image embedding, score the length of
// its change from the original's embedding along the prompt's unit direction. Unlike the cosine between the
// two, this rewards a larger change: with the cosine, the search settled on lightly altered photographs, and a
// blind reviewer preferred random recipes, which change more.
import { OPS, barelyChanged, isFlat, normalized, render } from "./effects.mjs";
import { randomNode, randomRecipe, randomRegion, round, tidy } from "./recipes.mjs";
import { dot, sub } from "./score.mjs";

const MAX_NODES = 8;
// Two pictures whose image embeddings have a cosine above this look alike: a candidate this close to the original
// reads as the photograph unchanged (a blind reviewer's judgment), and one this close to a pool entry as a duplicate.
const SIMILAR = 0.92;
// At most this many kept entries use any one operation: CLIP tells four tile shuffles in different colours apart, a
// viewer sees one idea four times. Blind reviewers preferred pools with this limit 3 to 2, one tie.
const SHARED = 3;

// A copy of the recipe with one or two random changes: a parameter, a strength, a node inserted, deleted,
// replaced, repeated or moved, a region added, moved or removed, or a stretch of another recipe spliced in.
function mutate(recipe, rnd, other) {
  const out = normalized(recipe);
  do change(out, rnd, other);
  while (rnd() < 0.35);
  const tidied = tidy(out);
  return tidied.length ? tidied : [randomNode(rnd)];
}

function change(r, rnd, other) {
  const at = Math.floor(rnd() * r.length), node = r[at], u = rnd();
  if (!node || (u < 0.12 && r.length < MAX_NODES)) r.splice(Math.floor(rnd() * (r.length + 1)), 0, randomNode(rnd));
  else if (u < 0.45) tweak(node, rnd);
  else if (u < 0.55) node.strength = round(Math.min(1, Math.max(OPS[node.op].least ?? 0.2, node.strength + gauss(rnd) * 0.2)));
  else if (u < 0.63 && r.length > 1) r.splice(at, 1);
  else if (u < 0.71) r[at] = node.region ? { ...randomNode(rnd), region: node.region } : randomNode(rnd);
  else if (u < 0.79 && r.length > 1) {
    const to = Math.floor(rnd() * r.length);
    r.splice(to, 0, r.splice(at, 1)[0]);
  } else if (u < 0.84 && r.length < MAX_NODES) {
    const copy = structuredClone(node);
    tweak(copy, rnd);
    r.splice(at + 1, 0, copy);
  } else if (u < 0.93) reshape(node, rnd);
  else if (other?.length) {
    const from = Math.floor(rnd() * other.length), n = 1 + Math.floor(rnd() * Math.min(3, other.length - from));
    r.splice(at, Math.floor(rnd() * 2), ...structuredClone(other.slice(from, from + n)));
    r.length = Math.min(r.length, MAX_NODES);
  } else tweak(node, rnd);
}

// One parameter moved by a Gaussian step of a sixth of its range, or an integer redrawn or stepped by one.
function tweak(node, rnd) {
  const { params, int = [] } = OPS[node.op], names = Object.keys(params);
  if (!names.length) return;
  const k = names[Math.floor(rnd() * names.length)], [min, max] = params[k], v = node.params[k];
  if (!int.includes(k)) node.params[k] = round(Math.min(max, Math.max(min, v + gauss(rnd) * (max - min) / 6)));
  else if (max - min > 50 || rnd() < 0.5) node.params[k] = min + Math.floor(rnd() * (max - min + 1));
  else node.params[k] = Math.min(max, Math.max(min, v + (rnd() < 0.5 ? -1 : 1)));
}

// A region added, nudged (moved, turned or its level shifted, for the shapes that have each) or removed (scaled by
// whole). Borders stay soft: small hard patches read as defects.
function reshape(node, rnd) {
  const g = node.region;
  if (!g) node.region = randomRegion(rnd);
  else if (rnd() < 0.3 * (OPS[node.op].whole ?? 1)) delete node.region;
  else {
    if (g.shape <= 4) {
      g.x = round(Math.min(1 - g.w, Math.max(0, g.x + gauss(rnd) * 0.1)));
      g.y = round(Math.min(1 - g.h, Math.max(0, g.y + gauss(rnd) * 0.1)));
    }
    if (g.shape === 2 || g.shape === 3) g.angle = round((g.angle + gauss(rnd) * 30 + 540) % 360 - 180);
    if (g.shape >= 5) g.level = round(g.level + gauss(rnd) * 0.1);  // normalized clamps it
  }
}

// Recipes to score next: random ones, or with a pinned entry, three in four children of the best entry that looks
// like it (itself or a refinement, see select), crossed with another entry now and then. At an equal budget, blind
// reviewers preferred the varied pool of random recipes to one bred automatically, which collapsed onto one look.
function propose(pool, n, rnd) {
  const pinned = pool.find((e) => e.pinned), any = () => pool[Math.floor(rnd() * pool.length)];
  const parent = pinned && pool.reduce((a, e) => (e.score > a.score && dot(e.emb, pinned.emb) >= SIMILAR ? e : a), pinned);
  return Array.from({ length: n }, () => (parent && rnd() < 0.75 ? mutate(parent.recipe, rnd, any().recipe) : randomRecipe(rnd)));
}

// An entry for a recipe and its unit image embedding. options: orig, the original's embedding, and direction.
export function entry(recipe, emb, { orig, direction }) {
  return { recipe, emb, score: dot(sub(emb, orig), direction) };
}

// One batch: `batch` proposed recipes rendered from the source, without those that render flat or barely changed,
// embedded (embed: images => unit vectors), and scored, without those whose embedding barely moved from the
// original's. Returns the new entries, unselected.
export async function step(pool, options) {
  const { source, embed, rnd, batch, orig } = options;
  const children = propose(pool, batch, rnd).flatMap((recipe) => {
    const out = render(source, recipe);
    return isFlat(out, source) || barelyChanged(out, source) ? [] : [{ recipe, out }];
  });
  const vecs = children.length ? await embed(children.map((c) => c.out)) : [];
  return children.flatMap(({ recipe }, k) => (dot(vecs[k], orig) > SIMILAR ? [] : [entry(recipe, vecs[k], options)]));
}

// The best entries, at most `size`, best first, skipping any that looks like one already kept. Before a pick, an
// entry using an operation that SHARED of them already use waits for places left over, so the pool favours variety
// but never shrinks (after a pick, it is meant to hold the pick's children, which share its operations). A pinned
// entry is always kept, and comes first. So is the best entry that looks like it alone and scores higher: a
// refinement of the pick, which its children then come from. Without it, about a fifth of the proposals after a
// pick were better versions of it, dropped as its duplicates.
export function select(entries, size) {
  const pinned = entries.find((e) => e.pinned), kept = pinned ? [pinned] : [], uses = new Map(), waiting = [];
  const alike = (e) => kept.filter((k) => dot(k.emb, e.emb) >= SIMILAR);
  let refined = false;
  for (const e of [...entries].sort((a, b) => b.score - a.score)) {
    if (kept.length >= size) break;
    if (e === pinned) continue;
    const ops = pinned ? [] : [...new Set(e.recipe.map((n) => n.op))];
    if (ops.some((op) => uses.get(op) >= SHARED)) { waiting.push(e); continue; }
    const like = alike(e);
    if (!like.length) { kept.push(e); for (const op of ops) uses.set(op, (uses.get(op) ?? 0) + 1); }
    else if (!refined && like.length === 1 && like[0] === pinned && e.score > pinned.score) { kept.push(e); refined = true; }
  }
  for (const e of waiting) {
    if (kept.length >= size) break;
    if (!alike(e).length) kept.push(e);
  }
  return kept.sort((a, b) => (b.pinned ? 1 : 0) - (a.pinned ? 1 : 0) || b.score - a.score);
}

const gauss = (rnd) => Math.sqrt(-2 * Math.log(1 - rnd())) * Math.cos(2 * Math.PI * rnd());
