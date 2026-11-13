// One tooltip for the whole window. Any element opts in with `use:tip={...}`; focus shows the
// same as hover. Content is usually a function: the tooltip calls it reactively, so what it
// shows stays current (ages tick, values change) while the pointer rests on the element.

export interface Tip {
  title: string;
  /** One line under the title, e.g. the state in words. */
  sub?: string;
  /** Plain-English explanation. */
  body?: string | string[];
  /** Label and value pairs. */
  rows?: [string, string][];
  /** Glossary entries to append: term and its meaning in the study's documents. */
  terms?: [string, string][];
  /** Where the value came from and how old it is. */
  source?: string;
}

export type TipInput = Tip | (() => Tip | null) | null | undefined;

class TipState {
  /** What to show: a tip, or a function that builds one from current data. */
  input = $state.raw<TipInput>(null);
  x = $state(0);
  y = $state(0);
  owner: Element | null = null;
}

export const tipState = new TipState();

export const resolveTip = (input: TipInput): Tip | null => (typeof input === "function" ? input() : (input ?? null));

export function showTip(content: TipInput, x: number, y: number, owner: Element | null = null) {
  tipState.input = content;
  tipState.x = x;
  tipState.y = y;
  tipState.owner = owner;
}

export function hideTip(owner: Element | null = null) {
  if (owner && tipState.owner !== owner) return;
  tipState.input = null;
  tipState.owner = null;
}

export function tip(node: Element, input: TipInput) {
  let current = input;
  const move = (e: Event) => {
    const p = e as PointerEvent;
    showTip(current, p.clientX, p.clientY, node);
  };
  const focus = () => {
    const r = node.getBoundingClientRect();
    showTip(current, r.left + r.width / 2, r.bottom, node);
  };
  const leave = () => hideTip(node);
  node.addEventListener("pointerenter", move);
  node.addEventListener("pointermove", move);
  node.addEventListener("pointerleave", leave);
  node.addEventListener("focus", focus);
  node.addEventListener("blur", leave);
  return {
    update(next: TipInput) {
      current = next;
      if (tipState.owner === node) tipState.input = next;
    },
    destroy() {
      leave();
      node.removeEventListener("pointerenter", move);
      node.removeEventListener("pointermove", move);
      node.removeEventListener("pointerleave", leave);
      node.removeEventListener("focus", focus);
      node.removeEventListener("blur", leave);
    },
  };
}
