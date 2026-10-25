### RePhraser

#### Install

`uv tool install git+https://github.com/35ABDA637EC99C/rephraser.git`

#### Usage:

As with most sane programs, `uv run rephraser.py -h` will list arguments it can take.

1. Get a corpus of sentences (or just a large text, like a text format ebook) to work with, I recommend choosing a file 
 from https://wortschatz.uni-leipzig.de/en/download/ and extracting the "sentences" file inside it (you should remove line
 numbers `sed -r 's/^[0-9]+[[:blank:]]*//' FILE`), as there should only be sentences in that file before giving it to RePhraser).

2. Then call RePhraser on it, piping the output to hashcat (I hope you have sufficient RAM, Markov models are memory-heavy to generate). 
  E.g. :
  `uv run rephraser.py --corpus sentences-eng_wikipedia_2016_1M --model ./wiki1M_model.keyvi | hashcat -m 1000 -O -r rephraser.rule ./NTLMHashfile`

3. Profit

For future runs, you can reuse the Markov Chain model RePhraser had to make (saved to the path specified with `--model`), call rephraser with just `--model ./wiki1M_model.keyvi`. Don't worry, the finished/compiled models are a fraction (as much as 1/40th) the size of the model while it is being constructed in markovify and then compiled into a keyvi dictionary.

#### Two paragraphs to use in Company Policy / Password Guidelines:

Passphrases that are easy to remember, but exceedingly difficult to crack, are unlikely or illogical phrases of five or more words.
Adding special characters or numbers or other complexity, although not required, is better done between or inside words
(dozens of possible positions) instead of only on the beginnings or ends of passphrases.

Combinations of thematically-unrelated words like:

"diabetic unicorn TRAPPED chicago tunnel"

Can be easy to remember if you say them to yourself "A diabetic-unicorn is trapped in a Chicago tunnel" and force an attacker to have
 a large dictionary with multiple types of words (cities, structures, fantasy creatures, verbs, medical conditions) to even have a chance of
 producing your password. Using words that wouldn't typically come after each-other in a sentence also forces an attacker to use brute force to guess
 combinations of words. Although it may seem complex, a logical phrase like "Docking With The International Space Station" (six fairly-uncommon words right? holy cow!)
 is much simpler to crack to an attacker using a natural language model, because "Docking" and "Space Station" are both thematically related
 and very likely to be together in a sentence. If an attacker is using "Docking" as the first word (through pure brute-force) to generate
 a phrase, the rest of the words are a likely, if not *the most likely* combination to follow it.

#### Dependencies:

- [Markovify](https://github.com/jsvine/markovify) for initial creation and compilation of Markov Chains
- [Keyvi](https://github.com/KeyviDev/keyvi) for efficient storage and shared-memory access by multiple processes
	 - [cmake](https://gitlab.kitware.com/cmake/cmake)  (apt package usually is `cmake`)
		 - [libboost](https://gitlab.kitware.com/cmake/cmake/issues/19402) (apt package usually is `libboost-all-dev`)
	 - [snappy](https://github.com/google/snappy) (apt package usually is `libsnappy-dev`)
- **Python 3.10+** 

#### Other file in this repo:
- `rephraser.rule` Rules to alter rephraser's output into different common patterns of capitalization and spaces within hashcat

