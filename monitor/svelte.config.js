import { vitePreprocess } from "@sveltejs/vite-plugin-svelte";

export default {
  preprocess: vitePreprocess(),
  compilerOptions: {
    runes: true,
    // Marks, chips and list rows carry tabindex on purpose: keyboard focus shows the same
    // explanation as hover, and those elements are not buttons.
    warningFilter: (warning) => warning.code !== "a11y_no_noninteractive_tabindex",
  },
};
