import spot
for s in ["p0 U p1", "p0 & !p0", "G(p0) & F(!p0)", "X p0"]:
    aut = spot.from_ltlf(s).translate('small', 'buchi', 'sbacc')
    print(s, "->", not aut.is_empty())