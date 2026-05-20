from search import search_by_text

results = search_by_text("a dog running on grass", top_k=5)
print(f"Got {len(results)} results:")
for meta, score in results:
    print(f"  {score:.3f}  {meta['filename']}")
