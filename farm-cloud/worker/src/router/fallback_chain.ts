// Default fallback construction for a plan node.
//
// Order is: next-best controller cards that overlap on at least one skill (by
// descending best skill confidence, with cost as the tie-breaker), then the
// classical-planner backend last. Classical is always present, even when the
// primary card already lists it in its own `fallbacks` field. This is the
// architectural commitment in DESIGN.md > Fallback chain.

import type { CapabilityCard } from "./prompt";

export const CLASSICAL_BACKEND_ID = "classical-planner";

function bestConfidenceForOverlap(
  primary: CapabilityCard,
  other: CapabilityCard,
): number {
  const primarySkills = new Set(primary.skills.map((s) => s.name));
  let best = -1;
  for (const skill of other.skills) {
    if (primarySkills.has(skill.name) && skill.confidence > best) {
      best = skill.confidence;
    }
  }
  return best;
}

export function build(
  primary: CapabilityCard,
  allCards: CapabilityCard[],
): string[] {
  const seen = new Set<string>([primary.id]);
  const candidates: { id: string; conf: number; cost: number }[] = [];
  for (const card of allCards) {
    if (seen.has(card.id)) continue;
    if (card.id === CLASSICAL_BACKEND_ID) continue;
    if (!card.roles.includes("controller")) continue;
    const conf = bestConfidenceForOverlap(primary, card);
    if (conf < 0) continue;
    candidates.push({
      id: card.id,
      conf,
      cost: card.cost_per_chunk_usd ?? Number.POSITIVE_INFINITY,
    });
    seen.add(card.id);
  }
  candidates.sort((a, b) => {
    if (b.conf !== a.conf) return b.conf - a.conf;
    return a.cost - b.cost;
  });
  const chain = candidates.map((c) => c.id);
  chain.push(CLASSICAL_BACKEND_ID);
  return chain;
}
