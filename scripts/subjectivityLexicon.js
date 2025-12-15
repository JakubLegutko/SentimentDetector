
// Basic Subjectivity Lexicon
// Compiled from common lists of subjective adjectives and adverbs.
// Used as a fallback when the transformer model is unavailable or to explain results.

const ENGLISH_LEXICON = [
  "amazing", "awesome", "awful", "bad", "beautiful", "best", "better", "boring", "brilliant",
  "creepy", "cute", "dangerous", "delicious", "difficult", "disgusting", "dull", "easy",
  "excellent", "exciting", "expensive", "fantastic", "favorite", "fun", "funny", "good",
  "gorgeous", "great", "guilty", "happy", "hard", "hate", "horrible", "important",
  "incredible", "interesting", "lovely", "magnificent", "marvelous", "nasty", "nice",
  "odd", "perfect", "pleasant", "poor", "powerful", "pretty", "remarkable", "sad",
  "scary", "serious", "silly", "simple", "strange", "stupid", "successful", "terrible",
  "terrifying", "ugly", "unbelievable", "unhappy", "useless", "wonderful", "worse", "worst",
  "wrong", "yummy", "adventurous", "aggressive", "agreeable", "ambitious", "amusing",
  "annoying", "anxious", "arrogant", "ashamed", "attractive", "average", "beneficial",
  "bewildered", "bizarre", "brainy", "brave", "breakable", "bright", "busy", "calm",
  "careful", "cautious", "charming", "cheerful", "clumsy", "comfortable", "concerned",
  "condemned", "confused", "cooperative", "courageous", "crazy", "creepy", "cruel",
  "curious", "dangerous", "defeated", "defiant", "delightful", "depressed", "determined",
  "different", "difficult", "disgusted", "distinct", "disturbed", "dizzy", "doubtful",
  "drab", "eager", "elated", "elegant", "embarrassed", "enchanting", "encouraging",
  "energetic", "enthusiastic", "envious", "evil", "excited", "expensive", "exuberant",
  "fair", "faithful", "famous", "fancy", "fierce", "filthy", "fine", "foolish", "fragile",
  "frail", "frantic", "friendly", "frightened", "gentle", "gifted", "glamorous", "gleaming",
  "glorious", "graceful", "grieving", "grotesque", "grumpy", "handsome", "healthy",
  "helpful", "helpless", "hilarious", "homeless", "homely", "hungry", "hurt", "ill",
  "impossible", "inexpensive", "innocent", "inquisitive", "itchy", "jealous", "jittery",
  "jolly", "joyous", "kind", "lazy", "light", "lively", "lonely", "long", "lucky", "misty",
  "modern", "motionless", "muddy", "mushy", "mysterious", "naughty", "nervous", "nutty",
  "obedience", "obnoxious", "old-fashioned", "open", "outrageous", "outstanding", "panicky",
  "plain", "poised", "precious", "prickly", "proud", "putrid", "puzzled", "quaint", "real",
  "relieved", "repulsive", "rich", "selfish", "shiny", "shy", "sleepy", "smiling", "smoggy",
  "sore", "sparkling", "splendid", "spotless", "stormy", "stupid", "super", "talented",
  "tame", "tasty", "tender", "tense", "thankful", "thoughtful", "thoughtless", "tired",
  "tough", "troubled", "ugliest", "uninterested", "unsightly", "unusual", "upset", "uptight",
  "vast", "victorious", "vivacious", "wandering", "weary", "wicked", "wide-eyed", "wild",
  "witty", "worried", "worrisome", "zany", "zealous"
];

const POLISH_LEXICON = [
  "wspaniały", "niesamowity", "okropny", "zły", "piękny", "najlepszy", "lepszy", "nudny",
  "genialny", "przerażający", "uroczy", "niebezpieczny", "pyszny", "trudny", "obrzydliwy",
  "tępy", "łatwy", "doskonały", "ekscytujący", "drogi", "fantastyczny", "ulubiony", "zabawny",
  "dobry", "olśniewający", "świetny", "winny", "szczęśliwy", "ciężki", "nienawidzić",
  "ważny", "niesłychany", "interesujący", "kochany", "wyśmienity", "cudowny", "wstrętny",
  "miły", "dziwny", "idealny", "przyjemny", "biedny", "potężny", "ładny", "niezwykły",
  "smutny", "straszny", "poważny", "głupi", "prosty", "dziwaczny", "skuteczny", "brzydki",
  "niewiarygodny", "nieszczęśliwy", "bezużyteczny", "gorszy", "najgorszy", "błędny",
  "pyszny", "agresywny", "ambitny", "atrakcyjny", "przeciętny", "korzystny", "dziwaczny",
  "odważny", "jasny", "zajęty", "spokojny", "ostrożny", "czarujący", "radosny", "niezdarny",
  "wygodny", "zmartwiony", "zdezorientowany", "szalony", "okrutny", "ciekawy", "pokonany",
  "zachwycający", "zdeterminowany", "inny", "zniesmaczony", "wątpliwy", "szary", "chętny",
  "elegancki", "zawstydzony", "zachęcający", "energetyczny", "entuzjastyczny", "zazdrosny",
  "zły", "podekscytowany", "sprawiedliwy", "wierny", "sławny", "luksusowy", "gwałtowny",
  "brudny", "delikatny", "utalentowany", "chwalebny", "pomocny", "bezradny", "komiczny",
  "bezdomny", "głodny", "zraniony", "chory", "niemożliwy", "tani", "niewinny", "dociekliwy",
  "swędzący", "nerwowy", "leniwy", "lekki", "żywy", "samotny", "długi", "szczęściarz",
  "mglisty", "nowoczesny", "nieruchomy", "błotnisty", "tajemniczy", "niegrzeczny",
  "posłuszny", "nieznośny", "staroświecki", "otwarty", "oburzający", "wybitny", "paniczny",
  "zwykły", "opanowany", "cenny", "kłujący", "dumny", "zgniły", "zaintrygowany", "prawdziwy",
  "odrażający", "bogaty", "samolubny", "lśniący", "nieśmiały", "śpiący", "uśmiechnięty",
  "bolesny", "błyszczący", "burzowy", "twardy", "zmartwiony", "najbrzydszy", "nieciekawy",
  "niecodzienny", "zdenerwowany", "spięty", "ogromny", "zwycięski", "błąkający się",
  "zmęczony", "nikczemny", "dziki", "dowcipny"
];

const SUBJECTIVITY_LEXICON = new Set([...ENGLISH_LEXICON, ...POLISH_LEXICON]);

window.SUBJECTIVITY_LEXICON = SUBJECTIVITY_LEXICON;
