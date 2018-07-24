 
import pandas as pd
import os
# ----

def createTypeTable():
	#%%%%%%
	#%%%%%% returns the name of the types, both long version
	#%%%%%% as the abbrevated version.
	#%%%%%%

	name_abbr 	= "ac, ba, bc, bg, cb, ck, db, ei, ga, \
gb, id, kh, ma, nb, sb, sp, tb, tg" 

	name_long 	= "acceptgiro, betaalautomaat, betalen contactloos, \
bankgiro_opdracht, crediteurenbetaling, \
Chipknip, diverse_boekingen, euro-incasso, \
geldautomaat_Euro, geldautomaat_VV, iDEAL, \
kashandeling, machtiging, NotaBox, salaris_betaling, \
spoedopdracht, eigen_rekening, telegiro"

	abbr_list 	= name_abbr.split(",")		
	long_list	= name_long.split(",")

	dt_abr = pd.DataFrame({'abbr_list':abbr_list,'long_list':long_list})

	return dt_abr


# ---- Create supermarkten data


def createSuperTable():
	#%%%%%%
	#%%%%%% returns the name of the types, both long version
	#%%%%%% as the abbrevated version.
	#%%%%%%

	super_long = "Agrimarkt,Albert Heijn,AH to go,Aldi, \
Attent,Attent Super op vakantie!, Boni,Coop,CoopCompact, \
Dagwinkel, Deen,Deka Markt,Dirk,EMTÉ Supermarkten, \
E-markt,Hoogvliet,Jan Linders,Jumbo, Kingsalmarkt,Lidl,\
MCD,Makro, M&M supermarkten,Nettorama,Pakgro, Picnic, \
Plus,Poiesz,Pryma,Sligro,Spar, Supercoop,Troefmarkt,Vomar, \
Boon, Albertheijn"
# now translate this
	supermarkt = list(map(str.strip,super_long.split(",")))
	dt_super = pd.DataFrame({'supermarkt':supermarkt})
	return(dt_super)
 

# ---- Create Shop data


def createShopTable():
  #%%%%%%
  
	shop_long = "Mediamarkt, Decathlon, Gamma, Ikea, Hema, Kruidvat, Gall.+Gall		"
	shop = list(map(str.strip,shop_long.split(",")))
	dt_shop = pd.DataFrame({'winkel':shop})
	return(dt_shop) 

# ---- Create city data


def createCityTable():
  #%%%%%%

	city_long = "Delft, Utrecht, Zaandam, Amsterdam, Amsterdam Zui, Rotterdam, 's-Gravenhage"
	city = list(map(str.strip,city_long.split(",")))
	dt_city = pd.DataFrame({'stad':city})
	return(dt_city)  


# --- Write data
location_data = "/home/charmmaria/Documents/data/Transacties/" 
os.chdir(location_data)
createTypeTable().to_csv("typeTable.csv",index = False)
createSuperTable().to_csv("marktTable.csv",index = False)
createShopTable().to_csv("shopTable.csv",index = False)
createCityTable().to_csv("cityTable.csv",index = False)

