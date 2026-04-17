import re
from grammar.grammar import Schema, Templates, Query, TaskFactory

# from grammar import Schema, Templates, Query, TaskFactory

PEOPLE_NAMES = [
    "John", "Mary", "Carla", "Bob", "Sam", "Alex", "Emma", "David", "Sarah", "Michael",
    "Lisa", "James", "Anna", "Daniel", "Sophie", "Chris", "Rachel", "Tom", "Nina", "Peter",
    "Laura", "George", "Mark", "Kate", "Henry", "Julia", "Oscar", "Mia", "Zoe", "Adam",
    "Eva", "Luke", "Nora", "Ryan", "Ella", "Owen", "Isla", "Jack", "Rose", "Liam",
    "Amy", "Noah", "Grace", "Ethan", "Clara", "Mason", "Chloe", "Logan", "Alice", "Lucas",
    "Diana", "Oliver", "Helen", "Elijah", "Vera", "Aiden", "Ruth", "Jackson", "Iris", "Sebastian",
    "Lydia", "Mateo", "Edith", "Theodore", "Agnes", "Hugo", "Cecilia", "Felix", "Miriam", "Adrian",
    "Sylvia", "Dominic", "Ingrid", "Vincent", "Petra", "Marcus", "Astrid", "Julian", "Selene", "Patrick",
    "Aurora", "Simon", "Freya", "Gabriel", "Thea", "Nathan", "Dora", "Aaron", "Esther", "Blake",
    "Naomi", "Victor", "Carmen", "Wesley", "Bianca", "Dennis", "Fatima", "Xavier", "Grant", "Jude",
    "Wyatt", "Caleb", "Levi", "Maya", "Ruby", "Cora", "Stella"
]
EELINGS = [
    "loves", "admires", "misses", "hates", "dislikes", "likes", "appreciates", "grieves", "mourns", "honors",
    "praises", "fears", "doubts", "resents", "shuns", "minds", "shames", "blames", "shocks", "thrills",
    "amazes", "alarms", "elates", "upsets", "annoys", "bores", "tires", "puzzles", "charms", "delights",
    "stuns", "awes", "moves", "touches", "repels", "disgusts", "appalls", "daunts", "cheers", "maddens",
    "enrages", "fumes", "riles", "piques", "goads", "vexes", "irks", "fretts", "panics", "scares",
    "frights", "dazes", "muses", "hopes", "longs", "yearns", "pines", "lusts", "relies", "braves",
    "dares", "grudges", "spites", "gloats", "triumphs", "rejoices", "exults", "glows", "beams", "smiles",
    "frowns", "glares", "sneers", "smirks", "winces", "quakes", "shivers", "glowers", "values", "prizes",
    "treasures", "idolizes", "worships", "pities", "envies", "comforts", "trusts", "scorns", "loathes", "detests",
    "abhors", "tolerates", "endures", "mallows", "jolts", "flatters", "recoils", "craves", "dreads", "reveres",
    "fancies", "covets", "relishes", "disdains", "mocks", "taunts", "chides", "scolds", "presents", "notices",
    "observes", "studies", "watches", "scans", "eyes", "views", "respects", "cherishes", "embraces", "rejects"
]

DOCTOR_NAMES = [
    "Dr. Smith", "Dr. Lee", "Dr. Patel", "Dr. Chen", "Dr. Brown", "Dr. Garcia", "Dr. Kim", "Dr. Wilson", "Dr. Nguyen", "Dr. Cohen",
    "Dr. Adams", "Dr. Baker", "Dr. Carter", "Dr. Davis", "Dr. Evans", "Dr. Foster", "Dr. Gray", "Dr. Hill", "Dr. Ivan", "Dr. Jones",
    "Dr. Kelly", "Dr. Lewis", "Dr. Moore", "Dr. Nash", "Dr. Owen", "Dr. Page", "Dr. Quinn", "Dr. Reed", "Dr. Scott", "Dr. Todd",
    "Dr. Uzun", "Dr. Vega", "Dr. Ward", "Dr. Xiao", "Dr. Yang", "Dr. Zane", "Dr. Abbas", "Dr. Bello", "Dr. Costa", "Dr. Devi",
    "Dr. Ebo", "Dr. Fang", "Dr. Gobi", "Dr. Hara", "Dr. Issa", "Dr. Jeng", "Dr. Kalu", "Dr. Lima", "Dr. Mori", "Dr. Neto",
    "Dr. Oka", "Dr. Pena", "Dr. Qadir", "Dr. Rosa", "Dr. Sato", "Dr. Tuan", "Dr. Udoh", "Dr. Vera", "Dr. Wang", "Dr. Xavi",
    "Dr. Yune", "Dr. Zidi", "Dr. Ahab", "Dr. Bond", "Dr. Cole", "Dr. Duke", "Dr. Earl", "Dr. Finn", "Dr. Galt", "Dr. Holt",
    "Dr. Ives", "Dr. Jude", "Dr. Kane", "Dr. Lord", "Dr. Moss", "Dr. Noel", "Dr. Pike", "Dr. Rhys", "Dr. Shay", "Dr. Tate",
    "Dr. Vale", "Dr. West", "Dr. York", "Dr. Zale", "Dr. Arif", "Dr. Bash", "Dr. Cruz", "Dr. Dorn", "Dr. Elms", "Dr. Frye",
    "Dr. Gage", "Dr. Hope", "Dr. Iron", "Dr. Jace", "Dr. Kent", "Dr. Lyle", "Dr. Monk", "Dr. Noon", "Dr. Ogle", "Dr. Park",
    "Dr. Ross", "Dr. Silva", "Dr. Tran", "Dr. Vance", "Dr. Wong", "Dr. Yates", "Dr. Zhu"
]

CONTAINERS = [
    "cup", "jug", "bottle", "glass", "mug", "bowl", "vase", "pitcher", "flask", "jar",
    "bucket", "urn", "trough", "crate", "case", "bin", "pack", "sack", "bag", "box",
    "drum", "tray", "silo", "cylinder", "beaker", "goblet", "chalice", "thermos", "decanter", "stein",
    "horn", "ampoule", "casket", "chest", "trunk", "coffer", "hopper", "skip", "scuttle", "brazier",
    "crucible", "purse", "wallet", "satchel", "knapsack", "backpack", "briefcase", "valise", "grip", "holdall",
    "quiver", "sheath", "holster", "scabbard", "folder", "envelope", "capsule", "pod", "shell", "hull",
    "nacelle", "gondola", "carboy", "firkin", "hogshead", "puncheon", "butt", "tun", "keg", "cask",
    "canteen", "bota", "cruet", "phial", "ewer", "carafe", "flagon", "pipkin", "skillet", "wok",
    "cauldron", "kettle", "boiler", "copper", "alembic", "retort", "samovar", "terrine", "tureen", "pail",
    "canister", "shaker", "tub", "vat", "vial", "tank", "basin", "barrel", "pouch", "tin",
    "hamper", "pot", "ampule", "flasket", "humidor", "reliquary"
]

HOUSEHOLD_ITEMS = [
    "egg", "fan", "tea", "engine", "plate", "gift", "wire", "watch", "cross", "boat",
    "game", "rose", "shell", "seed", "magnet", "suit", "ticket", "tie", "card", "brain",
    "fig", "wheel", "machine", "note", "drink", "bread", "camera", "bill", "chemical", "clock",
    "flower", "creature", "rock", "plant", "sheet", "leaf", "block", "newspaper", "disk", "boot",
    "medicine", "coffee", "book", "ball", "string", "fish", "crown", "branch", "phone", "plane",
    "apple", "bell", "brick", "document", "file", "bus", "drug", "computer", "mirror", "stone",
    "radio", "dress", "meat", "train", "bomb", "letter", "guitar", "hat", "map", "magazine",
    "coat", "television", "painting", "picture", "milk", "pipe", "ice", "key", "broom", "pillow",
    "blanket", "towel", "soap", "brush", "comb", "spoon", "fork", "knife", "lamp", "rug",
    "shifter", "hammer", "screw", "nail", "drill", "ladder", "basket", "mop", "shovel", "bucket",
    "toaster", "kettle", "dryer", "blender", "vacuum", "iron", "pan", "pot", "scissors", "needle", "thread"
]

LIQUIDS = [
    "beer", "tea", "soda", "water", "coffee", "juice", "milk", "wine", "cider", "cola",
    "champagne", "whiskey", "vodka", "rum", "gin", "punch", "tonic", "oil", "broth", "syrup",
    "gravy", "soup", "plasma", "ink", "blood", "sweat", "tears", "saliva", "venom", "poison",
    "antidote", "potion", "lotion", "cream", "gel", "paste", "sap", "resin", "nectar", "honey",
    "molasses", "vinegar", "brine", "bleach", "ammonia", "chlorine", "ethanol", "methanol", "diesel", "petrol",
    "gasoline", "kerosene", "paraffin", "naphtha", "benzene", "toluene", "acetone", "chloroform", "ether", "acid",
    "base", "lye", "solvent", "paint", "dye", "stain", "varnish", "lacquer", "enamel", "primer",
    "sealant", "glue", "adhesive", "cement", "grout", "mortar", "plaster", "stucco", "clay", "mud",
    "sludge", "slime", "ooze", "muck", "mire", "magma", "lava", "espresso", "latte", "mocha",
    "cappuccino", "macchiato", "frappe", "smoothie", "shake", "kefir", "bourbon", "brandy", "tequila", "absinthe",
    "cognac", "sake", "mead", "sanitizer", "shampoo", "conditioner"
]

HOUSEHOLD_LOCATIONS = [
    "kitchen", "library", "office", "park", "garage", "bedroom", "basement", "bathroom", "hallway", "attic",
    "closet", "pantry", "balcony", "porch", "garden", "yard", "patio", "terrace", "rooftop", "workshop",
    "cellar", "den", "nursery", "foyer", "lounge", "study", "gym", "sauna", "pool", "driveway",
    "shed", "greenhouse", "gazebo", "pergola", "deck", "loft", "mudroom", "laundry", "vault", "bunker",
    "shelter", "scullery", "larder", "bootroom", "cloakroom", "vestibule", "landing", "staircase", "corridor", "gallery",
    "atrium", "courtyard", "outhouse", "barn", "stable", "coop", "paddock", "orchard", "vineyard", "grove",
    "solarium", "alcove", "parlor", "drawing", "suite", "chamber", "anteroom", "ballroom", "belfry", "buttery",
    "cabin", "chancel", "crypt", "cupboard", "dormitory", "enclosure", "entryway", "garret", "hearth", "kitchenette",
    "lavatory", "mezzanine", "observatory", "passageway", "portico", "pulpit", "refectory", "salon", "sanctuary", "scaffold",
    "sewing", "storehouse", "veranda", "wardrobe", "utility", "archives", "armory", "boudoir", "dungeon", "infirmary",
    "playroom", "turret", "conservatory", "sunroom", "living", "dining"
]

COUNTRIES = [
    "USA", "Canada", "UK", "Australia", "India", "China", "Japan", "Germany", "France", "Israel",
    "Brazil", "Mexico", "Italy", "Spain", "Portugal", "Russia", "Norway", "Sweden", "Finland", "Denmark",
    "Poland", "Greece", "Turkey", "Egypt", "Argentina", "Chile", "Peru", "Colombia", "Venezuela", "Ecuador",
    "Bolivia", "Paraguay", "Uruguay", "Guyana", "Suriname", "Cuba", "Haiti", "Jamaica", "Bahamas", "Belize",
    "Guatemala", "Honduras", "Nicaragua", "Panama", "Iceland", "Ireland", "Belgium", "Luxembourg", "Switzerland", "Austria",
    "Czechia", "Slovakia", "Hungary", "Romania", "Bulgaria", "Serbia", "Croatia", "Bosnia", "Montenegro", "Albania",
    "Macedonia", "Malta", "Cyprus", "Syria", "Lebanon", "Jordan", "Iraq", "Iran", "Yemen", "Oman",
    "Qatar", "Kuwait", "Bahrain", "Georgia", "Armenia", "Azerbaijan", "Kazakhstan", "Uzbekistan", "Turkmenistan", "Tajikistan",
    "Kyrgyzstan", "Afghanistan", "Pakistan", "Nepal", "Bhutan", "Bangladesh", "Myanmar", "Thailand", "Laos", "Vietnam",
    "Cambodia", "Malaysia", "Singapore", "Indonesia", "Philippines", "Fiji", "Samoa", "Tonga", "Vanuatu", "Kiribati",
    "Nauru", "Tuvalu", "Palau", "Estonia", "Latvia", "Lithuania", "Morocco", "Algeria", "Tunisia", "New Zealand", "Netherlands"
]

MUSIC_GENRES = [
    "jazz", "classical", "rock", "blues", "folk", "electronic", "country", "pop", "funk", "metal",
    "rap", "techno", "house", "trance", "soul", "punk", "disco", "indie", "gospel", "ska",
    "opera", "latin", "dub", "ambient", "reggae", "salsa", "mambo", "tango", "samba", "bossa",
    "rumba", "cumbia", "bachata", "merengue", "calypso", "soca", "zouk", "rai", "afrobeat", "highlife",
    "juju", "kwaito", "bongo", "taarab", "dangdut", "bhangra", "qawwali", "carnatic", "hindustani", "gamelan",
    "gagaku", "minyo", "enka", "kpop", "jpop", "cpop", "mandopop", "cantopop", "trot", "mariachi",
    "ranchera", "norteno", "tejano", "conjunto", "banda", "corridos", "flamenco", "fado", "polka", "klezmer",
    "celtic", "bluegrass", "zydeco", "cajun", "motown", "barbershop", "choral", "acappella", "madrigal", "gregorian",
    "chant", "drone", "noise", "glitch", "chiptune", "synthwave", "vaporwave", "chillwave", "grunge", "emo",
    "screamo", "math", "shoegaze", "britpop", "grime", "dubstep", "garage", "trap", "drill", "lofi", "swing", "bebop"
]

MUSIC_INSTRUMENTS = [
    "piano", "guitar", "violin", "drums", "flute", "trumpet", "bass", "accordion", "organ", "tabla",
    "triangle", "recorder", "whistle", "bell", "horn", "pipe", "stick", "clap", "drum", "string",
    "block", "beat", "tam", "cello", "harp", "oboe", "tuba", "viola", "banjo", "sitar",
    "lute", "lyre", "fife", "gong", "cymbal", "kazoo", "zither", "oud", "koto", "cornet",
    "bugle", "fagott", "piccolo", "celesta", "marimba", "timpani", "bongo", "conga", "guiro", "claves",
    "cowbell", "cabasa", "shekere", "tambourine", "castanet", "maracas", "vibraphone", "xylophone", "glockenspiel", "ocarina",
    "didgeridoo", "bagpipes", "harmonica", "concertina", "melodica", "synthesizer", "theremin", "keyboard", "sampler", "sequencer",
    "spinet", "harpsichord", "clavichord", "cittern", "mandolin", "ukulele", "balalaika", "bouzouki", "dulcimer", "psaltery",
    "rebec", "vielle", "hurdygurdy", "shawm", "crumhorn", "bombard", "serpent", "sackbut", "cornett", "gemshorn",
    "duduk", "pipa", "erhu", "shamisen", "gusli", "kantele", "veena", "sarod", "santur", "rebab",
    "darbuka", "daf", "shaker", "kora", "valiha", "cajon", "djembe", "kalimba"
]

SUBSTANCES = [
    "serum", "plasma", "enzyme", "protein", "acid", "base", "solvent", "dye", "water", "salt",
    "sugar", "starch", "oil", "sand", "clay", "stone", "metal", "glass", "wax", "ash",
    "smoke", "steam", "gas", "ink", "glue", "paint", "lead", "lime", "coal", "iron",
    "gold", "silver", "zinc", "copper", "brass", "bronze", "steel", "wool", "silk", "cotton",
    "linen", "nylon", "plastic", "rubber", "latex", "foam", "pulp", "paper", "cork", "wood",
    "bark", "leaf", "moss", "dirt", "dust", "grit", "gravel", "mica", "talc", "quartz",
    "flint", "slate", "marble", "chalk", "borax", "alum", "soda", "urea", "resin", "amber",
    "tallow", "suet", "lard", "pork", "beef", "fish", "egg", "milk", "whey", "curd",
    "yeast", "mold", "rust", "scum", "silt", "peat", "tar", "pitch", "asphalt", "bitumen",
    "char", "coke", "soot", "brine", "alkali", "oxide", "niter", "slag", "magma", "graphite",
    "diamond", "sapphire", "ruby", "emerald", "jade"
]

CHEMICALS = [
    "ethanol", "chlorine", "ammonia", "sulfur", "carbon", "oxygen", "nitrogen", "helium", "neon", "lithium",
    "sodium", "potassium", "calcium", "magnesium", "copper", "zinc", "iron", "nickel", "silver", "gold",
    "platinum", "uranium", "tin", "lead", "hydrogen", "boron", "fluorine", "aluminum", "silicon", "argon",
    "scandium", "titanium", "vanadium", "chromium", "cobalt", "gallium", "arsenic", "selenium", "bromine", "krypton",
    "rubidium", "yttrium", "niobium", "ruthenium", "rhodium", "cadmium", "indium", "antimony", "tellurium", "iodine",
    "xenon", "cesium", "barium", "lanthanum", "cerium", "samarium", "europium", "terbium", "holmium", "erbium",
    "thulium", "lutetium", "hafnium", "tantalum", "tungsten", "rhenium", "osmium", "iridium", "mercury", "thallium",
    "bismuth", "polonium", "astatine", "radon", "francium", "radium", "actinium", "thorium", "neptunium", "plutonium",
    "curium", "berkelium", "fermium", "nobelium", "hassium", "meitnerium", "dubnium", "bohrium", "mendelevium", "lawrencium",
    "tennessine", "oganesson", "moscovium", "nihonium", "flerovium", "germanium", "phosphorus"
]

APPARATUSES = [
    "flask", "funnel", "jar", "tube", "dish", "balance", "burner", "clamp", "stand", "hood",
    "probe", "tray", "slide", "scoop", "bath", "filter", "cylinder", "rod", "oven", "dryer",
    "press", "mortar", "cap", "beaker", "pipette", "burette", "syringe", "needle", "forceps", "scalpel",
    "spatula", "pestle", "crucible", "tongs", "tweezers", "magnifier", "lens", "prism", "mirror", "laser",
    "diode", "sensor", "meter", "gauge", "dial", "switch", "relay", "motor", "pump", "valve",
    "hose", "pipe", "joint", "adapter", "plug", "socket", "wire", "cable", "cord", "battery",
    "panel", "screen", "monitor", "keypad", "mouse", "stylus", "printer", "plotter", "scanner", "camera",
    "tripod", "mount", "stage", "frame", "rack", "cabinet", "drawer", "shelf", "bench", "stool",
    "cart", "trolley", "ladder", "crane", "hoist", "jack", "winch", "lathe", "mill", "drill",
    "saw", "grinder", "sander", "buffer", "welder", "torch", "magnet", "hemostat", "centrifuge", "spectrometer",
    "microscope", "telescope", "petri-dish", "caliper", "voltmeter", "ammeter", "ohmmeter", "galvanometer", "barometer", "hygrometer",
    "anemometer", "incubator"
]

VEHICLES = [
    "car", "bus", "truck", "bike", "scooter", "van", "jeep", "tram", "train", "taxi",
    "cart", "coach", "sled", "boat", "ship", "yacht", "canoe", "raft", "ferry", "subway",
    "submarine", "buggy", "cab", "moped", "lorry", "wagon", "coupe", "sedan", "limo", "rover",
    "tank", "dozer", "crane", "glider", "rocket", "shuttle", "balloon", "blimp", "drone", "jet",
    "hover", "skiff", "dinghy", "dory", "barge", "tug", "liner", "cutter", "frigate", "cruiser",
    "sloop", "ketch", "yawl", "punt", "sampan", "junk", "dhow", "kayak", "pirogue", "coracle",
    "umiak", "proa", "catamaran", "trimaran", "monohull", "hydrofoil", "surfboard", "skis", "skates", "trike",
    "quad", "atv", "utv", "suv", "rv", "camper", "trailer", "tractor", "combine", "forklift",
    "backhoe", "grader", "skidder", "paver", "roller", "ambulance", "prowler", "roadster", "biplane", "monoplane",
    "triplane", "gondola", "funicular", "teleferic", "skylift", "clippers", "steamer", "galley", "trireme", "unicycle",
    "tricycle", "skateboard", "longboard", "snowmobile", "jetski", "hang-glider", "parachute", "paddleboard", "trawler", "tugboat",
    "rickshaw", "segway"
]

DESTINATIONS = [
    "school", "market", "airport", "office", "station", "garage", "park", "mall", "harbor", "stadium",
    "hotel", "theater", "cinema", "museum", "zoo", "aquarium", "beach", "temple", "church", "mosque",
    "castle", "palace", "plaza", "arena", "campus", "library", "hospital", "clinic", "pharmacy", "bank",
    "court", "prison", "base", "fort", "camp", "farm", "ranch", "vineyard", "orchard", "forest",
    "jungle", "desert", "island", "mountain", "valley", "canyon", "cavern", "lake", "river", "ocean",
    "sea", "bay", "gulf", "port", "dock", "pier", "wharf", "quay", "bridge", "tunnel",
    "tower", "spire", "dome", "shrine", "monument", "statue", "fountain", "square", "parkway", "highway",
    "road", "street", "avenue", "lane", "path", "trail", "track", "route", "way", "spot",
    "place", "site", "point", "zone", "area", "region", "state", "city", "town", "village",
    "hamlet", "suburb", "district", "province", "parish", "bakery", "cafe", "diner", "bistro"
]

SPORTS = [
    "soccer", "tennis", "rugby", "baseball", "cricket", "hockey", "volleyball", "golf", "boxing", "wrestling",
    "cycling", "skiing", "skating", "surfing", "sailing", "fencing", "shooting", "swimming", "running", "climbing",
    "polo", "chess", "marathon", "sprint", "hurdles", "relay", "vault", "dive", "rowing", "judo",
    "karate", "taekwondo", "wushu", "sambo", "sumo", "kickboxing", "archery", "darts", "snooker", "billiards",
    "curling", "bobsleigh", "skeleton", "luge", "biathlon", "triathlon", "pentathlon", "decathlon", "heptathlon", "gymnastics",
    "parkour", "motocross", "karting", "rally", "badminton", "lacrosse", "handball", "softball", "squash", "racquetball",
    "pickleball", "netball", "rounders", "hurling", "camogie", "shinty", "croquet", "petanque", "bocce", "bowls",
    "quidditch", "dodgeball", "paintball", "airsoft", "skateboarding", "snowboarding", "wakeboarding", "kiteboarding", "windsurfing", "paddleboarding",
    "kayaking", "canoeing", "rafting", "canyoning", "spelunking", "orienteering", "aerobics", "crossfit", "bodybuilding", "powerlifting",
    "weightlifting", "yoga", "pilates", "futsal", "kabaddi", "sepaktakraw", "kickball"
]

VENUES = [
    "stadium", "court", "arena", "gym", "track", "hall", "park", "field", "pitch", "pool",
    "ring", "dojo", "grounds", "course", "range", "greens", "circus", "terrace", "oval", "dome",
    "complex", "plaza", "center", "studio", "gallery", "museum", "library", "theater", "cinema", "ballroom",
    "dancehall", "nightclub", "discotheque", "casino", "resort", "hotel", "motel", "hostel", "inn", "tavern",
    "bistro", "cafe", "restaurant", "canteen", "cafeteria", "pantry", "kitchen", "workshop", "factory", "warehouse",
    "depot", "terminal", "station", "airport", "hangar", "garage", "basement", "attic", "loft", "cellar",
    "vault", "bunker", "shelter", "tent", "cabin", "cottage", "house", "villa", "palace", "castle",
    "fort", "base", "camp", "office", "bank", "prison", "school", "campus", "clinic", "hospital",
    "pharmacy", "church", "temple", "mosque", "shrine", "cathedral", "chapel", "monastery", "convent", "abbey",
    "priory", "mission", "hermitage", "retreat", "sanctuary", "asylum", "orphanage", "nursery", "kindergarten", "academy",
    "pavilion"
]

SPACE_OBJECTS = [
    "planet", "comet", "asteroid", "galaxy", "star", "belt", "meteor", "moon", "sun", "cluster",
    "dwarf", "nova", "constellation", "void", "halo", "jet", "core", "disk", "ring", "flare",
    "cloud", "rock", "dust", "ice", "gas", "pulsar", "quasar", "nebula", "blazar", "magnetar",
    "supernova", "exoplanet", "supergiant", "hypernova", "kilonova", "protostar", "magnetosphere", "plasmasphere", "heliosphere", "chromosphere",
    "photosphere", "exosphere", "thermosphere", "mesosphere", "stratosphere", "troposphere", "ionosphere", "magnetopause", "heliopause", "bowshock",
    "barycenter", "pericenter", "apocenter", "perihelion", "aphelion", "perigee", "apogee", "periastron", "apastron", "syzygy",
    "eclipse", "transit", "occultation", "conjunction", "opposition", "quadrature", "solstice", "equinox", "precession", "nutation",
    "libration", "parallax", "magnitude", "luminosity", "albedo", "spectrum", "redshift", "blueshift", "parsec", "lightyear",
    "zenith", "nadir", "azimuth", "altitude", "declination", "ascension", "ecliptic", "equator", "meridian", "horizon",
    "zodiac", "asterism", "supercluster", "filament", "supervoid", "singularity", "ergosphere", "photon", "boson", "wormhole",
    "blackhole", "meteoroid"
]

SPACE_INSTRUMENTS = [
    "telescope", "camera", "detector", "sensor", "radar", "finder", "microscope", "scanner", "tracker", "monitor",
    "scope", "lens", "mirror", "prism", "array", "dish", "probe", "rover", "satellite", "antenna",
    "gyro", "compass", "gauge", "altimeter", "lidar", "sonar", "magnetometer", "gravimeter", "seismometer", "thermometer",
    "barometer", "hygrometer", "anemometer", "pyranometer", "pyrheliometer", "actinometer", "ceilometer", "transmissometer", "nephelometer", "dosimeter",
    "spectrograph", "coronagraph", "interferometer", "bolometer", "polarimeter", "fluorometer", "spectroscope", "spectroheliograph", "spectrohelioscope", "heliometer",
    "micrometer", "chronometer", "sextant", "octant", "quadrant", "astrolabe", "gnomon", "sundial", "theodolite", "transit",
    "collimator", "photoguide", "autoguider", "eyepiece", "focuser", "mount", "tripod", "pier", "dome", "shutter",
    "filter", "grating", "slit", "mask", "chopper", "modulator", "demodulator", "amplifier", "digitizer", "processor",
    "computer", "recorder", "transmitter", "receiver", "transceiver", "transponder", "encoder", "decoder", "multiplexer", "demultiplexer",
    "oscillator", "synthesizer", "timer", "clock", "counter", "logic", "gate", "buffer", "driver", "bus"
]

EQUIPMENT = [
    "drill", "hammer", "saw", "wrench", "pliers", "pump", "valve", "sensor", "meter", "gauge",
    "motor", "engine", "gear", "belt", "chain", "crane", "hoist", "jack", "winch", "lathe",
    "grinder", "sander", "buffer", "welder", "torch", "magnet", "hemostat", "clamp", "vise", "press",
    "anvil", "forge", "kiln", "furnace", "boiler", "chiller", "radiator", "vent", "fan", "blower",
    "filter", "strainer", "nozzle", "sprinkler", "hose", "pipe", "tube", "duct", "adapter", "coupler",
    "fitting", "joint", "seal", "gasket", "bearing", "bushing", "pulley", "sprocket", "clutch", "brake",
    "piston", "shaft", "crank", "cam", "lever", "link", "spring", "damper", "mount", "frame",
    "chassis", "housing", "casing", "panel", "plate", "bracket", "shelf", "rack", "cabinet", "drawer",
    "bin", "tray", "pallet", "trolley", "cart", "wagon", "truck", "forklift", "loader", "digger",
    "scraper", "grader", "roller", "paver", "mixer", "crusher", "shredder", "baler", "mower", "tiller",
    "chisel", "file", "mallet", "screwdriver", "awl", "router", "caliper"
]

BOX_VARS = [f"{i:02d}" for i in range(100)]

SCHEMA_FILLING_LIQUIDS = Schema(
    name="filling_liquids",
    items={
        "Person": PEOPLE_NAMES,
        "Container": CONTAINERS,
        "Liquid": LIQUIDS,
    },
    templates=Templates(
        # prefix="{Person_list} are working at a busy restaurant. To complete an order, ",
        prefix="Some people are working at a busy restaurant. To complete an order, ",
        definitions={
            "row_default": "{Person} fills a {Container} with {Liquid}",
            "ordering_012": "{Person} fills a {Container} with {Liquid}",
            "ordering_102": "a {Container} was filled by {Person} with {Liquid}",
            "ordering_120": "a {Container} was filled with {Liquid} by {Person}",
            "ordering_021": "{Person} pours {Liquid} into a {Container}",
            "ordering_201": "{Liquid} was poured by {Person} into a {Container}",
            "ordering_210": "{Liquid} was poured into a {Container} by {Person}",
        },
        queries={
            "Q:Container_Person A:Liquid": Query(
                question="Respond in one word, only the answer and nothing else: What does {Person} believe the {Container} contains?",
                answer_category="Liquid",
            ),
            "default": Query(
                question="Respond in one word, only the answer and nothing else: What does {Person} believe the {Container} contains?",
                answer_category="Liquid",
            ),
            "Q:Liquid_Person A:Container": Query(
                question="Respond in one word, only the answer and nothing else: What did {Person} fill with {Liquid}?",
                answer_category="Container",
            ),
            "Q:Person A:Container": Query(
                question="Respond in one word, only the answer and nothing else: What container did {Person} fill?",
                answer_category="Container",
            ),
            "Q:Container_Liquid A:Person": Query(
                question="Respond in one word, only the answer and nothing else: Who filled the {Container} with {Liquid}?",
                answer_category="Person",
            ),
        },
        capitalize_first_clause=False,
    ),
    max_new_tokens=5,
    checker=lambda neural, causal: causal.lower().strip() in neural.lower().strip(),
    matchers=[
        lambda s: re.match(f"^ ?({'|'.join(PEOPLE_NAMES)})$", s) is not None,
        lambda s: re.match(f"^ ?({'|'.join(CONTAINERS)})$", s) is not None,
        lambda s: re.match(f"^ ?({'|'.join(LIQUIDS)})$", s) is not None,
    ],
)

SCHEMA_PEOPLE_AND_OBJECTS = Schema(
    name="people_and_objects",
    items={"Person": PEOPLE_NAMES, "Object": HOUSEHOLD_ITEMS, "Location": HOUSEHOLD_LOCATIONS},
    templates=Templates(
        prefix="At home, ",
        definitions={
            "row_default": "{Person} put the {Object} in the {Location}",
            "row_reversed": "the {Object} was put in the {Location} by {Person}",
            "col_default": "{Person_list} put the {Object_list} in the {Location_list}, respectively.",
            "ordering_012": "{Person} put the {Object} in the {Location}",
        },
        queries={
            "Q:Object_Person A:Location": Query(
                question="Respond in one word, only the answer and nothing else: Where did {Person} put the {Object}?",
                answer_category="Location",
            ),
            "Q:Person A:Object": Query(
                question="Respond in one word, only the answer and nothing else: Which object did {Person} put?",
                answer_category="Object",
            ),
            "Q:Location_Person A:Object": Query(
                question="Respond in one word, only the answer and nothing else: What did {Person} put in the {Location}?",
                answer_category="Object",
            ),
            "Q:Location_Object A:Person": Query(
                question="Respond in one word, only the answer and nothing else: Who put the {Object} in the {Location}?",
                answer_category="Person",
            ),
            "default": Query(
                question="Respond in one word, only the answer and nothing else: Where did {Person} put the {Object}?",
                answer_category="Location",
            ),
        },
    ),
    matchers=[
        lambda s: re.match(f"^ ?({'|'.join(PEOPLE_NAMES)})$", s) is not None,
        lambda s: re.match(f"^ ?({'|'.join(HOUSEHOLD_ITEMS)})$", s) is not None,
        lambda s: re.match(f"^ ?({'|'.join(HOUSEHOLD_LOCATIONS)})$", s) is not None,
    ],
)

SCHEMA_PROGRAMMING_PEOPLE_DICT = Schema(
    name="programming_people_dict",
    items={"VariableName": BOX_VARS, "Name": PEOPLE_NAMES, "Country": COUNTRIES},
    templates=Templates(
        definitions={
            "row_default": '{VariableName} = {{"name": " {Name}", "country": " {Country}"}}',
            "ordering_012": '{VariableName} = {{"name": " {Name}", "country": " {Country}"}}',
        },
        queries={
            "default": Query(
                question='Respond in one word, only the answer and nothing else: What is the country in variable {VariableName} where name="{Name}"?',
                answer_category="Country",
            ),
            "Q:Name_VariableName A:Country": Query(
                question='Respond in one word, only the answer and nothing else: What is the country in variable {VariableName} where name="{Name}"?',
                answer_category="Country",
            ),
            "Q:VariableName A:Name": Query(
                question="Respond in one word, only the answer and nothing else: What is the name in variable {VariableName}?",
                answer_category="Name",
            ),
            "Q:Country_VariableName A:Name": Query(
                question='Respond in one word, only the answer and nothing else: What is the name in variable {VariableName} where country="{Country}"?',
                answer_category="Name",
            ),
            "Q:Country_Name A:VariableName": Query(
                question='Respond in one word, only the answer and nothing else: What is the variable name where country="{Country}" and name="{Name}"?',
                answer_category="VariableName",
            ),
        },
        capitalize_first_clause=False,
        prefix="The following are dictionary variables in Python: ",
    ),
    matchers=[
        lambda s: re.match(f"^ ?({'|'.join(BOX_VARS)})$", s) is not None,
        lambda s: re.match(f"^ ?({'|'.join(PEOPLE_NAMES)})$", s) is not None,
        lambda s: re.match(f"^ ?({'|'.join(COUNTRIES)})$", s) is not None,
    ],
)

SCHEMA_MUSIC_PERFORMANCE = Schema(
    name="music_performance",
    items={
        "Musician": PEOPLE_NAMES,
        "Genre": MUSIC_GENRES,
        "Instrument": MUSIC_INSTRUMENTS,
    },
    templates=Templates(
        prefix="At the music festival, ",
        definitions={
            "row_default": "{Musician} performed {Genre} music on the {Instrument}",
            "ordering_012": "{Musician} performed {Genre} music on the {Instrument}",
        },
        queries={
            "Q:Genre_Musician A:Instrument": Query(
                question="Respond in one word, only the answer and nothing else: What did {Musician} play {Genre} music on?",
                answer_category="Instrument",
            ),
            "Q:Musician A:Genre": Query(
                question="Respond in one word, only the answer and nothing else: What music did {Musician} play?",
                answer_category="Genre",
            ),
            "Q:Instrument_Musician A:Genre": Query(
                question="Respond in one word, only the answer and nothing else: What music did {Musician} play on the {Instrument}?",
                answer_category="Genre",
            ),
            "Q:Genre_Instrument A:Musician": Query(
                question="Respond in one word, only the answer and nothing else: What musician played {Genre} music on the {Instrument}?",
                answer_category="Musician",
            ),
        },
        capitalize_first_clause=False,
    ),
    checker=lambda neural, causal: causal.lower().strip() in neural.lower().strip(),
    max_new_tokens=3,
    matchers=[
        lambda s: re.match(f"^ ?({'|'.join(PEOPLE_NAMES)})$", s) is not None,
        lambda s: re.match(f"^ ?({'|'.join(MUSIC_GENRES)})$", s) is not None,
        lambda s: re.match(f"^ ?({'|'.join(MUSIC_INSTRUMENTS)})$", s) is not None,
    ],
)

SCHEMA_LAB_EXPERIMENTS = Schema(
    name="lab_experiments",
    items={
        "Scientist": PEOPLE_NAMES,
        "Substance": SUBSTANCES,
        "Equipment": EQUIPMENT,
    },
    templates=Templates(
        prefix="In a biology laboratory experiment, ",
        definitions={
            "row_default": "{Scientist} placed the {Substance} in a {Equipment}",
            "ordering_012": "{Scientist} placed the {Substance} in a {Equipment}",
        },
        queries={
            "Q:Scientist_Substance A:Equipment": Query(
                question="Respond in one word, only the answer and nothing else: What did {Scientist} place the {Substance} in?",
                answer_category="Equipment",
            ),
            "Q:Scientist A:Substance": Query(
                question="Respond in one word, only the answer and nothing else: What did {Scientist} place?",
                answer_category="Substance",
            ),
            "Q:Equipment_Scientist A:Substance": Query(
                question="Respond in one word, only the answer and nothing else: What did {Scientist} place in a {Equipment}?",
                answer_category="Substance",
            ),
            "Q:Equipment_Substance A:Scientist": Query(
                question="Respond in one word, only the answer and nothing else: Who placed the {Substance} in a {Equipment}?",
                answer_category="Scientist",
            ),
        },
        capitalize_first_clause=False,
    ),
    matchers=[
        lambda s: re.match(f"^ ?({'|'.join(PEOPLE_NAMES)})$", s) is not None,
        lambda s: re.match(f"^ ?({'|'.join(SUBSTANCES)})$", s) is not None,
        lambda s: re.match(f"^ ?({'|'.join(EQUIPMENT)})$", s) is not None,
    ],
    max_new_tokens=5,
    checker=lambda neural, causal: causal.lower().strip().split()[-1] in neural.lower().strip(),
)

SCHEMA_CHEMISTRY_EXPERIMENTS = Schema(
    name="chemistry_experiments",
    items={
        "Chemist": PEOPLE_NAMES,
        "Chemical": CHEMICALS,
        "Apparatus": APPARATUSES,
    },
    templates=Templates(
        prefix="In a chemistry laboratory experiment, ",
        definitions={
            "row_default": "{Chemist} added the {Chemical} to a {Apparatus}",
            "ordering_012": "{Chemist} added the {Chemical} to a {Apparatus}",
        },
        queries={
            "Q:Chemical_Chemist A:Apparatus": Query(
                question="Respond in one word, only the answer and nothing else: Which apparatus did {Chemist} use for the {Chemical}?",
                answer_category="Apparatus",
            ),
            "Q:Chemist A:Chemical": Query(
                question="Respond in one word, only the answer and nothing else: What chemical did {Chemist} add?",
                answer_category="Chemical",
            ),
            "Q:Apparatus_Chemist A:Chemical": Query(
                question="Respond in one word, only the answer and nothing else: What chemical did {Chemist} add to a {Apparatus}?",
                answer_category="Chemical",
            ),
            "Q:Apparatus_Chemical A:Chemist": Query(
                question="Respond in one word, only the answer and nothing else: Who added the {Chemical} to a {Apparatus}?",
                answer_category="Chemist",
            ),
        },
        capitalize_first_clause=False,
    ),
    matchers=[
        lambda s: re.match(f"^ ?({'|'.join(PEOPLE_NAMES)})$", s) is not None,
        lambda s: re.match(f"^ ?({'|'.join(CHEMICALS)})$", s) is not None,
        lambda s: re.match(f"^ ?({'|'.join(APPARATUSES)})$", s) is not None,
    ],
    max_new_tokens=5,
    checker=lambda neural, causal: causal.lower().strip().split()[-1] in neural.lower().strip(),
)

SCHEMA_TRANSPORTATION = Schema(
    name="transportation",
    items={
        "Driver": PEOPLE_NAMES,
        "Vehicle": VEHICLES,
        "Destination": DESTINATIONS,
    },
    templates=Templates(
        prefix="In a city transport system, ",
        definitions={
            "row_default": "{Driver} drove the {Vehicle} to the {Destination}",
            "ordering_012": "{Driver} drove the {Vehicle} to the {Destination}",
        },
        queries={
            "Q:Driver_Vehicle A:Destination": Query(
                question="Respond in one word, only the answer and nothing else: Where did {Driver} drive the {Vehicle}?",
                answer_category="Destination",
            ),
            "Q:Driver A:Vehicle": Query(
                question="Respond in one word, only the answer and nothing else: What vehicle did {Driver} drive?",
                answer_category="Vehicle",
            ),
            "Q:Destination_Driver A:Vehicle": Query(
                question="Respond in one word, only the answer and nothing else: What vehicle did {Driver} drive to the {Destination}?",
                answer_category="Vehicle",
            ),
            "Q:Destination_Vehicle A:Driver": Query(
                question="Respond in one word, only the answer and nothing else: Who drove the {Vehicle} to the {Destination}?",
                answer_category="Driver",
            ),
        },
        capitalize_first_clause=False,
    ),
    matchers=[
        lambda s: re.match(f"^ ?({'|'.join(PEOPLE_NAMES)})$", s) is not None,
        lambda s: re.match(f"^ ?({'|'.join(VEHICLES)})$", s) is not None,
        lambda s: re.match(f"^ ?({'|'.join(DESTINATIONS)})$", s) is not None,
    ],
    max_new_tokens=3,
    checker=lambda n, c: c.lower().strip() in n.lower().strip(),
)

SCHEMA_SPORTS_EVENTS = Schema(
    name="sports_events",
    items={
        "Athlete": PEOPLE_NAMES,
        "Sport": SPORTS,
        "Venue": VENUES,
    },
    templates=Templates(
        prefix="In a sports competition, ",
        definitions={
            "row_default": "{Athlete} played {Sport} at the {Venue}",
            "ordering_012": "{Athlete} played {Sport} at the {Venue}",
        },
        queries={
            "Q:Athlete_Sport A:Venue": Query(
                question="Respond in one word, only the answer and nothing else: Where did {Athlete} play {Sport}?",
                answer_category="Venue",
            ),
            "Q:Athlete A:Sport": Query(
                question="Respond in one word, only the answer and nothing else: What sport did {Athlete} play?",
                answer_category="Sport",
            ),
            "Q:Athlete_Venue A:Sport": Query(
                question="Respond in one word, only the answer and nothing else: What sport did {Athlete} play at the {Venue}?",
                answer_category="Sport",
            ),
            "Q:Sport_Venue A:Athlete": Query(
                question="Respond in one word, only the answer and nothing else: Who played {Sport} at the {Venue}?",
                answer_category="Athlete",
            ),
        },
        capitalize_first_clause=False,
    ),
    matchers=[
        lambda s: re.match(f"^ ?({'|'.join(PEOPLE_NAMES)})$", s) is not None,
        lambda s: re.match(f"^ ?({'|'.join(SPORTS)})$", s) is not None,
        lambda s: re.match(f"^ ?({'|'.join(VENUES)})$", s) is not None,
    ],
    max_new_tokens=3,
    checker=lambda n, c: c.lower().strip() in n.lower().strip(),
)

SCHEMA_SPACE_OBSERVATIONS = Schema(
    name="space_observations",
    items={
        "Astronomer": PEOPLE_NAMES,
        "Object": SPACE_OBJECTS,
        "Instrument": SPACE_INSTRUMENTS,
    },
    templates=Templates(
        prefix="During an astronomy study, ",
        definitions={
            "row_default": "{Astronomer} observed a {Object} with a {Instrument}",
            "ordering_012": "{Astronomer} observed a {Object} with a {Instrument}",
        },
        queries={
            "Q:Astronomer_Object A:Instrument": Query(
                question="Respond in one word, only the answer and nothing else: Which instrument did {Astronomer} use to observe the {Object}?",
                answer_category="Instrument",
            ),
            "Q:Astronomer A:Object": Query(
                question="Respond in one word, only the answer and nothing else: What did {Astronomer} observe?",
                answer_category="Object",
            ),
            "Q:Astronomer_Instrument A:Object": Query(
                question="Respond in one word, only the answer and nothing else: What did {Astronomer} observe with a {Instrument}?",
                answer_category="Object",
            ),
            "Q:Instrument_Object A:Astronomer": Query(
                question="Respond in one word, only the answer and nothing else: Who observed a {Object} with a {Instrument}?",
                answer_category="Astronomer",
            ),
        },
        capitalize_first_clause=False,
    ),
    max_new_tokens=5,
    matchers=[
        lambda s: re.match(f"^ ?({'|'.join(PEOPLE_NAMES)})$", s) is not None,
        lambda s: re.match(f"^ ?({'|'.join(SPACE_OBJECTS)})$", s) is not None,
        lambda s: re.match(f"^ ?({'|'.join(SPACE_INSTRUMENTS)})$", s) is not None,
    ],
    checker=lambda neural, causal: causal.lower().strip().split()[-1] in neural.lower().strip(),
)

SCHEMA_BOXES = Schema(
    name="boxes",
    items={"Object": HOUSEHOLD_ITEMS, "Box": [x.upper() for x in BOX_VARS]},
    templates=Templates(
        prefix="",
        definitions={
            "row_default": "the {Object} is in Box {Box}",
            "ordering_01": "the {Object} is in Box {Box}",
        },
        queries={
            "Q:Box A:Object": Query(
                question="Respond in one word, only the answer and nothing else: What does Box {Box} contain?",
                answer_category="Object",
            ),
            "Q:Object A:Box": Query(
                question="Respond in one word, only the answer and nothing else: Which box is the {Object} in? Box",
                answer_category="Box",
            ),
        },
        capitalize_first_clause=True,
    ),
    max_new_tokens=3,
    checker=lambda neural, causal: causal
    in re.search("(Box )?([A-Z])", neural.strip()).group(2).strip(),  # Checker for when querying the letters
    # checker=lambda neural, causal: causal.strip().lower() in neural.strip().lower(), # Checker for when querying the items
    matchers=[
        lambda s: re.match(f"^ ?({'|'.join(HOUSEHOLD_ITEMS)})$", s) is not None,
        lambda s: re.match(r"^ ?\d{2}$", s.strip()) is not None,
    ],
)

if __name__ == "__main__":
    import pandas as pd

    schemas = [
        SCHEMA_FILLING_LIQUIDS,
        SCHEMA_MUSIC_PERFORMANCE,
        SCHEMA_PEOPLE_AND_OBJECTS,
        SCHEMA_PROGRAMMING_PEOPLE_DICT,
        SCHEMA_LAB_EXPERIMENTS,
        SCHEMA_CHEMISTRY_EXPERIMENTS,
        SCHEMA_TRANSPORTATION,
        SCHEMA_SPORTS_EVENTS,
        SCHEMA_SPACE_OBSERVATIONS,
        SCHEMA_BOXES,
    ]

    rows = []
    for schema in schemas:
        task_factory = TaskFactory(schema)
        task_instance = task_factory.create_task_instance(num_instances=2)
        task = task_instance.generate_task(definition_key="row_default", query_instance_idx=0)
        final_form = f"{task['context']} {task['question']}"
        rows.append({"Name": schema.name, "Task": final_form, "Answer": task["answer"]})

    df = pd.DataFrame(rows, columns=["Name", "Task", "Answer"])
    df.to_csv("/tmp/schemas.csv", index=False)
