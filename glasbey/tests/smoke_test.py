import glasbey

# Create a categorical palette with 15 colours
glasbey.create_palette(palette_size=15)
# Create a muted palette with 12 colours
glasbey.create_palette(palette_size=12, lightness_bounds=(20, 40), chroma_bounds=(40, 50))

