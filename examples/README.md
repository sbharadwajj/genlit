# Example inputs

The shipped JSONs point at the bundled images under `images/` so you can try
the model immediately after `pip install -e .` — no setup required.

## Run from the repo root

```bash
python -m genlit.inference --mode single --img_json examples/single.json --output_dir out/
python -m genlit.inference --mode multi  --img_json examples/multi.json  --output_dir out/
```

The JSON paths are relative to the repo root, so run from there.

## Provided examples

| File                          | Mode   | Description                                 |
|-------------------------------|--------|---------------------------------------------|
| `images/bowl.png`             | single | Ceramic bowl on a dark surface (512×512)    |
| `images/multi_scene.png`      | multi  | Multi-object tabletop scene (768×512)       |

## Using your own images

Edit the JSON to point at your files (absolute or repo-relative paths both work):

```json
{
  "my_object": "/absolute/path/to/your/image.jpg",
  "another":   "examples/images/another.png"
}
```

For best results, use images whose subject is approximately centered and well-lit.
The single model is trained on 512×512 inputs; multi is trained on 640×448. Other
aspects/sizes are resized at load time.
