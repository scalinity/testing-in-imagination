<script lang="ts">
  import type { GlyphName } from "./status";

  // A 12-unit glyph that works both in HTML and nested inside another SVG (pass x and y there).
  let { name, size = 11, x, y }: { name: GlyphName; size?: number; x?: number; y?: number } = $props();
</script>

{#if name !== "none"}
  <svg class="glyph" viewBox="0 0 12 12" width={size} height={size} {x} {y} aria-hidden="true">
    {#if name === "ring"}
      <circle cx="6" cy="6" r="3.4" />
    {:else if name === "queued"}
      <circle cx="6" cy="6" r="3.9" />
      <circle cx="6" cy="6" r="1.2" class="solid" />
    {:else if name === "dot"}
      <circle cx="6" cy="6" r="3.2" class="solid" />
    {:else if name === "check"}
      <path d="M2.4 6.4 5 9l4.6-5.6" />
    {:else if name === "cross"}
      <path d="M3 3l6 6M9 3l-6 6" />
    {:else if name === "pause"}
      <path d="M4.3 2.8v6.4M7.7 2.8v6.4" />
    {:else if name === "retry"}
      <path d="M9.6 6.3A3.6 3.6 0 1 1 8.2 3.2" />
      <path d="M8.6 1.2v2.3H6.3" />
    {:else if name === "skip"}
      <path d="M2.5 6h7" />
    {:else if name === "clock"}
      <circle cx="6" cy="6" r="4.3" />
      <path d="M6 3.6V6l1.7 1.1" />
    {/if}
  </svg>
{/if}

<style>
  .glyph {
    display: inline-block;
    flex: none;
    fill: none;
    stroke: currentColor;
    stroke-width: 1.5;
    stroke-linecap: round;
    stroke-linejoin: round;
    vertical-align: -1px;
    overflow: visible;
  }
  .solid {
    fill: currentColor;
    stroke: none;
  }
</style>
