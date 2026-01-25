from ai import ai_embeds

embeds = ai_embeds()

embed = embeds.text_embed("cat")

print(embed)
print(type(embed))