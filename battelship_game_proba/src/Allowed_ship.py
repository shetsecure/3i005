from Ship import Ship

allowed_ships_dict = dict() # values are Ship objects

allowed_ships_dict['porte_avions'] = Ship('porte_avions', 5, 10) # length, color
allowed_ships_dict['croiseur'] = Ship('croiseur', 4, 20) # length, color
allowed_ships_dict['contre_torpilleurs'] = Ship('contre_torpilleurs', 3, 30) # length, color
allowed_ships_dict['sous_marin'] = Ship('sous_marin', 3, 40) # length, color
allowed_ships_dict['torpilleur'] = Ship('torpilleur', 2, 50) # length, color