// Writes one static HTML page per loop example, for build_static.py.
// Run with Quarto's bundled Deno: quarto run visualizations/export-static.js <outdir>
const here = new URL(".", import.meta.url);
await import(new URL("loop-widgets.js", here).href);
const out = Deno.args[0];
for (const name of globalThis.LoopWidgets.names) {
  await Deno.writeTextFile(`${out}/loop-${name}.html`, globalThis.LoopWidgets.staticHTML(name));
}
console.log(globalThis.LoopWidgets.names.join(" "));
