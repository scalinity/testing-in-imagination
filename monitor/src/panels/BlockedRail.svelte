<script lang="ts">
  import Glyph from "../lib/Glyph.svelte";
  import { openBlocked } from "../lib/derive";
  import { clusterSource, fileSource } from "../lib/sources";
  import { store } from "../lib/store.svelte";
  import { terms } from "../lib/study";
  import { tip } from "../lib/tip.svelte";

  const items = $derived(openBlocked(store.remote, store.docs));
</script>

{#if items.length}
  <section class="rail" aria-label="Open blocked items">
    <span class="lead"><Glyph name="pause" size={12} /> Waits on you</span>
    <ul>
      {#each items as b (b.id)}
        <li
          tabindex="0"
          use:tip={() => ({
            title: `${b.id}: ${b.title}`,
            sub: b.opened ? `Open since ${b.opened}.` : "Open.",
            rows: [
              ["What waits", b.waits || "not stated"],
              ["Why", b.why || "not stated"],
              ["Your one step", b.request || "not stated"],
              ["Meanwhile", b.meanwhile || "not stated"],
            ],
            terms: terms("blocked"),
            source: b.source.includes("status.json") && !b.source.includes("BLOCKED.md and") && !b.source.startsWith("BLOCKED.md") ? clusterSource() : `${b.source}. ${fileSource(store.docs?.blocked_file)}`,
          })}
        >
          <span class="id code">{b.id}</span>
          <span class="title">{b.title}</span>
          {#if b.waits}<span class="waits">{b.waits}</span>{/if}
        </li>
      {/each}
    </ul>
  </section>
{/if}

<style>
  .rail {
    flex: none;
    display: flex;
    align-items: baseline;
    gap: 14px;
    padding: 7px 14px;
    border-bottom: 1px solid var(--line);
    background: color-mix(in srgb, var(--wait) 11%, var(--ground));
    box-shadow: inset 4px 0 0 var(--wait);
  }
  .lead {
    flex: none;
    font-weight: 700;
    color: var(--wait-text);
    white-space: nowrap;
  }
  ul {
    display: grid;
    gap: 3px;
    margin: 0;
    padding: 0;
    list-style: none;
    min-width: 0;
  }
  li {
    display: flex;
    align-items: baseline;
    gap: 10px;
    min-width: 0;
  }
  .id {
    flex: none;
    color: var(--ink);
    font-weight: 600;
  }
  .title {
    flex: none;
    font-weight: 600;
    color: var(--ink);
  }
  .waits {
    color: var(--ink-2);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    min-width: 0;
  }
</style>
