# The Little Guy: collected outputs, run 1 (base-u96x256-swiglu-r10)

Sampled during and after pretraining (2026-08-12 → 08-14), 20L/H768 uniform
SwiGLU MoE, 2.44B tokens with the cookbook SPT mix. Temperatures noted where
they matter. Preserved for the writeup and for posterity.

## The capital of France: a bildungsroman

- **step 1000:** "One of France's most iconic landmarks is the Pulté in
  France."
- **step 2500:** "Nehruh (France), the capital of France, is the national
  capital of France. It was founded in 1731 by the patroness of the French
  Revolutionary Congress, John Bruyn."
- **step 3966 (stable ckpt):** "The capital of France is the French word
  'Paris.' ... Paris is also home to approximately 1.8 million peo—" ✅
  *(and, unprompted: France is "notorious for its strong, spacious buildings,
  vibrant food" — he cannot not mention the food)*
- **post-SFT-from-base (broken):** "France is the capital of France, and it's
  the capital of the French capital. The capital is divided into two main
  cities, Paris and Lyon, each of which is a major city in France."

## Units of measure, invented

- "**½ teaspoon [Brunoise](./Beurre_Monté__The_Workhorse_Sauce.md#brunoise),
  or double cream**" — a knife cut used as a quantity, *with a citation to an
  actual French Laundry chapter*.
- "1¼ cups [Chocolate Sauce](./Basics.md#levesecs10); **or store-bought
  Scones**" — the professional "or store-bought" substitution template,
  minus the concept of equivalence.
- "1 cup (8 cups) chopped fresh parsley"
- "1½ cups minus 1/3 cup granulated sugar" *(this one is real cookbook idiom
  and he nailed it)*
- The epub bleed: "[skinned hazelnuts](./Chil%5F9780307958181%5Fepub%5F..."
  — verbatim internal hyperlink from the Julia Child ebook, ISBN and all.

## Technique, approximately

- "Stir the brine as the salt **decomposes**." (step 1500)
- Cover the brine and refrigerate "**for 1-2 minutes**." (step 1000)
- "**Discard any liquid that has fallen into the pot.**" (final, roast
  chicken)
- Roast chicken, subsection two: "**For the Filling:** 2 cups whole milk,
  2 cups heavy cream, 8 ounces chicken or turkey bones, chopped."
- Brine progression: honey-forward improv (1000) → salt+water identified
  (1500) → "2 cups water and 1 cup of salt" (2500, seasons with courage) →
  "2 cups water and **1 tsp** salt" (3966, overcorrected; bracketing like a
  line cook learning to season).

## Roux studies

- step 1000: "add a high-gloss roux to meatless meals... a higher-gloss roux
  (or higher-rising roux)"
- step 1500: "you have **enough flour to cover your base**" *(flour arrives!)*
- step 2500: "Making a base sauce is a traditional method of preparing stews"
  *(conceptually adjacent! roux IS a sauce base!)*
- final, temp 0.8: "Making a **blanco** (a grain of wheat) is a simple
  process that involves: **Honing**: To produce a roux—more nutritious and
  rich in nutrients."

## Brisket

- final, temp 0.8, on resting: "it's best to wait at least **24 hours** for
  it to settle down and become tender."
- final, temp 0.5, on resting: "**1 to 2 hours**" ✅ *(the sampling
  temperature lesson in one before/after)*
- final, temp 0.5, on smoking: "The key to smoking a brisket is to keep it
  moist" / (variable twin): "to be able to see the internal temperature" —
  both defensible!
- final, temp 0.8, on smoking: "eliminate all the bad elements that may have
  accumulated during the smoking process" *(garbled, but this is genuinely
  Franklin's clean-smoke gospel through a blender)*

## The Dave Arnold impressions (Cooking Issues probe)

- "the trick with a stock is to eliminate base stock. What if the base stock
  has been shifted to a different base stock? Preloaded base stock is to
  remove the base stock and add the base stock. The total stock is again
  removed."
- "30% and 30% are fine, but the stock is probably higher in price. 30% is
  great, and 30% is fine."
- (temp 1.0) "You might know it as the 'tongue in cheek' if you think your
  neighbor has a sweet tooth."

## The broken SFT (skipped midtraining; special tokens were noise)

- "How long should I rest a brisket?" → "AQC ⏎ 3 hours ⏎ 3 hours ⏎ 3 hours"
  ×44 *(the answer is roughly right; he just cannot stop saying it)*
- "Who are you?" → "1/2/3/3/3/3/3/3/3..." *(asked to introduce himself, he
  counted)*
- Quiz format discovered, content not: "**Answer:** **1 to 2 hours**
  **Explanation:** **Explanation:** **Explanation:** **Explanation:**
  **Explanation:** **Explanation:**" *(knows the answer; cannot say why —
  six section headers with nothing underneath)*
- Hollandaise diagnosis: "I made a new hollandaise sauce, and I added in the
  eggs... I also added in the flour and the lemon juice and the lemon zest
  and then added in the butter. I also added..." *(asked what went wrong,
  described doing it again, differently wrong)*

## Findings hiding in the jokes

- Register arrives before ingredients, ingredients before procedures,
  procedures before facts, facts before binding. (Probe strip: steps
  1000 → 1500 → 2500 → 3966 → final.)
- Templates transfer without semantics ("or store-bought X"), citations
  transfer without referents (Brunoise), notation leaks verbatim (epub
  links) — data cleaning strips images but should also strip text links.
- Sampling temperature is worth ~1500 steps of apparent capability at this
  scale: 24 hours at 0.8 vs the correct 1–2 hours at 0.5, same weights.
