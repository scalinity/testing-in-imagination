<script lang="ts">
  import { resolveTip, tipState } from "./tip.svelte";

  let width = $state(0);
  let height = $state(0);

  // Built inside a derived value, so it re-runs whenever the data it reads changes.
  const t = $derived(resolveTip(tipState.input));
  const paragraphs = $derived(!t?.body ? [] : Array.isArray(t.body) ? t.body : [t.body]);
  // Keep the box inside the window: prefer below and right of the pointer, flip when it would spill.
  const left = $derived(Math.max(8, tipState.x + 14 + width > window.innerWidth - 8 ? tipState.x - width - 12 : tipState.x + 14));
  const top = $derived(Math.max(8, tipState.y + 16 + height > window.innerHeight - 8 ? Math.max(8, tipState.y - height - 12) : tipState.y + 16));
</script>

{#if t}
  <div class="tip" role="tooltip" style:left="{left}px" style:top="{top}px" bind:clientWidth={width} bind:clientHeight={height}>
    <p class="title">{t.title}</p>
    {#if t.sub}<p class="sub">{t.sub}</p>{/if}
    {#each paragraphs as p}<p class="body">{p}</p>{/each}
    {#if t.rows?.length}
      <dl class="rows">
        {#each t.rows as [k, v]}
          <dt>{k}</dt>
          <dd>{v}</dd>
        {/each}
      </dl>
    {/if}
    {#if t.terms?.length}
      <dl class="terms">
        {#each t.terms as [term, meaning]}
          <div><dt>{term}</dt><dd>{meaning}</dd></div>
        {/each}
      </dl>
    {/if}
    {#if t.source}<p class="source">{t.source}</p>{/if}
  </div>
{/if}

<style>
  .tip {
    position: fixed;
    z-index: 100;
    max-width: 400px;
    min-width: 180px;
    padding: 10px 12px;
    background: var(--raised);
    border: 1px solid var(--line-2);
    border-radius: 6px;
    box-shadow: var(--shadow);
    pointer-events: none;
    font-size: var(--fs-2);
    line-height: 1.4;
    color: var(--ink-2);
  }
  .title {
    color: var(--ink);
    font-weight: 600;
    font-size: var(--fs-3);
  }
  .sub {
    color: var(--ink);
    margin-top: 1px;
  }
  .body {
    margin-top: 6px;
  }
  .rows {
    display: grid;
    grid-template-columns: max-content 1fr;
    gap: 2px 12px;
    margin: 8px 0 0;
  }
  .rows dt {
    color: var(--ink-3);
  }
  .rows dd {
    margin: 0;
    color: var(--ink);
    overflow-wrap: anywhere;
  }
  .terms {
    margin: 8px 0 0;
    padding-top: 7px;
    border-top: 1px solid var(--line);
    display: grid;
    gap: 4px;
  }
  .terms div {
    display: block;
  }
  .terms dt {
    display: inline;
    color: var(--ink);
    font-weight: 600;
  }
  .terms dt::after {
    content: ": ";
    font-weight: 400;
  }
  .terms dd {
    display: inline;
    margin: 0;
  }
  .source {
    margin-top: 8px;
    padding-top: 7px;
    border-top: 1px solid var(--line);
    color: var(--ink-3);
    font-size: var(--fs-1);
    overflow-wrap: anywhere;
  }
</style>
