"""
build_dataset.py  -  TakeMeter dataset assembler for r/hiphopheads.

Authors the labeled corpus as in-line Python lists (one post per line, hand-written
to vary length / slang / punctuation so the model can't key on a template), then
writes takemeter_dataset.csv with columns: text, label, notes.

Labels:
  critique  - structured evaluative argument about the music itself, citing a
              specific identifiable element (a bar, a beat switch, a producer, a
              track placement, a flow change). The reasoning survives removing the
              opinion framing.
  hot_take  - a bold, confident evaluative claim (ranking / GOAT / overrated)
              asserted without supporting musical evidence.
  stan      - an emotional fan reaction (hype, devotion, anticipation, letdown)
              where the feeling is the point and there is little/no argument.

Run:  python build_dataset.py
"""

import csv

# ----------------------------------------------------------------------------
# CRITIQUE  - cites a specific, identifiable musical element as the reason
# ----------------------------------------------------------------------------
critique = [
    "The sequencing on this is what carries it. Putting the two ambient interludes back to back at tracks 6 and 7 kills the momentum the first half built up.",
    "Kendrick's pocket on the second verse is unreal here, he's rapping a half beat behind the snare the whole time so it feels lazy and urgent at once.",
    "Production wise the 808s are mixed way too hot, they swallow the snare on like four of these songs and it makes the drums feel flat.",
    "What makes the bridge work is the key change going into the last chorus, it lifts the whole thing a third and that's why the outro hits so hard.",
    "Earl's whole appeal is the internal rhyme density, he'll stack three multis inside one bar so the line keeps folding back on itself.",
    "The sample flip is the move here, they pitched the original vocal up and chopped it on the off beat so it sounds like a totally new melody.",
    "Freddie Gibbs over Madlib works because Gibbs raps on top of the beat while Madlib's loops drag slightly behind, the friction is the texture.",
    "I think the mixing buries the bass guitar, you only really hear it on the bridge of track 9 and it's the best part of the record.",
    "Tyler's chord choices are what set him apart, those jazzy maj7 voicings under the hook give it that warmth no one else is getting.",
    "The reason the verse lands is the breath control, he goes 16 bars without a clear pause so the tension just keeps stacking.",
    "MF DOOM's rhyme scheme here ignores the bar lines completely, the punchline lands in the middle of the next bar and that's the whole joke.",
    "Cole's flow switch at the 1:40 mark is the highlight, he goes from triplets to a straight double time and the beat drops out to let it breathe.",
    "The drums on this are programmed too quantized, there's no swing so the whole thing feels stiff compared to the looser stuff he did before.",
    "JID's diction is the standout, he enunciates every consonant so even at double time you catch every word, which is rare at that speed.",
    "The way they layered the background vocals in thirds on the hook is why it feels so full, it's a choir trick on a rap song.",
    "Mach-Hommy's mixing is intentionally lo-fi but the vocal is always centered and dry so the bars cut through the murk, that's a deliberate choice.",
    "The beat switch at track 4 mirrors the lyric, the moment he says he's leaving the city the sample drops out and a cleaner loop comes in.",
    "Denzel's hardcore tracks work because the BPM sits around 140 which is faster than most rap, so the aggression is baked into the tempo.",
    "Pusha's whole thing is the specificity, he names the gram weight and the car trim so the boast feels reported instead of imagined.",
    "The reason the album drags in the back half is runtime, six of the last eight songs are over four minutes with no structural change.",
    "Andre's verse is built on enjambment, almost no line ends where the bar ends so the meaning keeps spilling into the next line.",
    "The Alchemist loop here never resolves, it loops on an unresolved chord so you keep waiting for a landing that never comes, that's the tension.",
    "Vince Staples raps quiet on purpose, he stays at a conversational volume so the violent content lands colder, the restraint is the point.",
    "The hook melody only uses three notes but the rhythm changes every other bar, that's why it's catchy without being repetitive.",
    "Little Simz's verse rides the string section instead of the drums, she phrases against the violins so it feels orchestral and rapped at once.",
    "The mastering is too loud, everything is slammed against the ceiling so there's no dynamic range and the quiet songs hit as hard as the loud ones.",
    "Black Thought's breath control is the lesson here, that ten minute freestyle has maybe four audible inhales and the cadence never slips.",
    "The reason the feature outshines the host is contrast, the guest comes in on a lower register so his verse sounds heavier by comparison.",
    "Madlib's drums are off grid on purpose, he plays them in by hand so the kick lands a hair late and that human drag is the whole vibe.",
    "The second half of the verse switches rhyme schemes from AABB to an internal scheme and that's exactly when the energy picks up.",
    "Kendrick uses three distinct voices across this track and each one is a different character, the pitch shifts are narrative not just texture.",
    "The bassline walks chromatically under the hook which is why the chorus feels like it's constantly moving forward even though the drums stay static.",
    "Gibbs lands the punchline on the and of four every time, so the joke always hits right before the bar resets, that timing is deliberate.",
    "The album's transitions are seamless because every track is in a related key, so the crossfades never clash, that's real sequencing work.",
    "Earl's verses got shorter and denser after this project, he's saying in eight bars what used to take sixteen and the compression is the growth.",
    "The reverb on the snare is the signature, it's a long plate reverb that tails into the next bar so the drums feel like they're in a big room.",
    "Tyler's bridge modulates up a whole step for the final chorus which is a pop songwriting move dropped into a rap album, that's why it soars.",
    "Danny Brown's pitched up delivery on this is a choice, the higher register makes the bleak lyrics sound manic instead of sad, tone and content clash on purpose.",
    "The reason the verse feels claustrophobic is the mix, the vocal is doubled and panned hard so it surrounds you instead of sitting center.",
    "Mick Jenkins phrases everything as questions on this track, the rising intonation at the end of each line keeps you off balance the whole verse.",
    "The drums don't come in until the 90 second mark and that long intro is why the drop feels earned when it finally lands.",
    "Roc Marciano raps with almost no hook structure, the verses just run continuously, which is why his stuff feels like prose more than songs.",
    "The melody on the chorus borrows the interval from the sampled gospel record, so it feels familiar before you've ever heard it, smart interpolation.",
    "Aesop Rock's vocabulary density means you need the lyrics sheet, but the payoff is the imagery is precise, every odd word is doing real work.",
    "The transition from the trap section to the soul section at 2:10 works because they kept the same tempo, so the switch feels like a reveal not a jolt.",
    "Noname raps behind the beat consistently which makes her sound conversational, like she's talking over the music rather than performing on it.",
    "The reason the outro hits is the strings drop out one section at a time, so the arrangement thins to just the kick and the vocal, that fade is composed.",
    "Westside Gunn's ad libs are percussion, the boom boom and the gunshots land on the rests in the drum pattern so they fill the gaps rhythmically.",
    "The verse uses the same rhyme sound for twelve straight bars which should get boring but he keeps changing the syllable count so the cadence stays alive.",
    "Kendrick's DAMN works backwards because the tracklist is reversible, the last song's beat resolves into the first song's intro, that's structural intent.",
    "The bass and the kick are sidechained hard so the bass ducks every time the kick hits, that pumping is why the low end feels so rhythmic.",
    "Joey Bada$$ phrases like a 90s rapper on purpose, the laid back behind the beat delivery is a deliberate throwback to the boom bap era he's referencing.",
    "The hook is doubled an octave up and that octave layer is mixed quiet so you feel it more than hear it, which is why the chorus sounds bigger than the verse.",
    "Quelle Chris leaves dead air in his verses, the pauses are part of the rhythm, he's using silence as a percussion element which most rappers won't risk.",
    "The reason the project feels cohesive is the same drum kit runs through all twelve tracks, the consistent sound palette ties it together more than the rapping does.",
    "Rapsody's verse builds by adding one syllable per bar to the cadence, so the flow accelerates without the tempo changing, that's a controlled escalation.",
    "The sample is slowed to half speed which drops it an octave and that's why the soul vocal sounds so mournful, the pitch manipulation does the emotional work.",
    "Open Mike Eagle's deadpan delivery is the joke setup, he says the absurd lines completely flat so the humor comes from the contrast with the tone.",
    "The snare is tuned to the key of the song which is why the drums never fight the melody, that's a producer detail almost nobody bothers with.",
    "Schoolboy Q rides the hi hats on this, his triplet flow locks to the hat pattern exactly so the vocal becomes part of the percussion.",
    "The bridge strips back to just piano and vocal and that vulnerability in the arrangement is why the emotional turn in the lyric actually registers.",
    "Billy Woods structures verses around recurring images rather than a narrative, the same objects keep returning so the meaning accrues instead of progressing.",
    "The reason the feature verse is better than the rest of the song is the guest matches the beat's swing while the host fights it the whole time.",
    "Kendrick's m.A.A.d city has two beats stitched together and the seam at the halfway switch is the most replayed moment because the contrast is so stark.",
    "The chorus uses a call and response between the lead and the doubled vocal, that back and forth is why it feels communal even though it's one rapper.",
    "Ka almost whispers his bars over beatless loops, removing the drums forces you to follow the words, that's the whole design of his sound.",
    "The 808 glides between notes on the hook instead of staying static, that pitch slide is a southern production signature and it's why the low end feels melodic.",
    "Lupe stacks a triple entendre in the second verse where the same line reads three ways, and the third reading reframes the first two, that's the craft.",
    "The mixing keeps the vocal slightly louder than is fashionable right now, which is why you can actually parse the lyrics, a deliberate anti trend choice.",
    "The drums on the back half switch from a boom bap loop to a live drummer and the human timing variance is what gives the closer its loose, jammy feel.",
]

# ----------------------------------------------------------------------------
# HOT_TAKE  - bold confident verdict / ranking, no supporting musical evidence
# ----------------------------------------------------------------------------
hot_take = [
    "Kendrick is the only rapper of his generation that will be remembered in 50 years, everyone else is filler.",
    "Drake has never made a good album front to back and people are too scared to say it.",
    "MF DOOM is the most overrated rapper in the history of the genre, there I said it.",
    "Tyler peaked on Igor and everything since has been a downgrade, full stop.",
    "Cole is a top 3 rapper alive and the people who disagree just don't get it.",
    "Trap killed real hip hop and we are never getting the golden era back.",
    "Travis Scott is an industry plant who can't actually rap, the production carries him completely.",
    "Nas hasn't been relevant since Illmatic and the stan delusion around him is insane.",
    "Future is more influential than Jay-Z and it's not even a debate at this point.",
    "Every Griselda project sounds exactly the same and the hype is pure cosplay nostalgia.",
    "Kanye is the greatest artist of the century, no rapper has come close to his impact.",
    "Lil Wayne in his prime would body any rapper out today and it wouldn't be close.",
    "JID is the most talented rapper under 30 and nobody else is even in the conversation.",
    "Eminem's technical skill is overrated, fast rapping isn't the same as good rapping.",
    "Playboi Carti changed the genre more than anyone in the last decade, period.",
    "Mac Miller would have been the best of his generation if he were still here, easily.",
    "Andre 3000 is the most naturally gifted rapper ever and it's barely an argument.",
    "Westside hip hop has been dead since 2015 and nobody wants to admit it.",
    "Denzel Curry is criminally underrated and in ten years people will call him a legend.",
    "Pop Smoke was carried entirely by the UK drill producers, the talent was never his.",
    "Kendrick's worst album is better than most rappers' best, that's just facts.",
    "Lupe Fiasco is smarter than every rapper out and the industry buried him for it.",
    "Tyler the Creator is the most important artist of his generation by a mile.",
    "Boom bap fans are stuck in 1996 and refuse to admit the genre evolved past them.",
    "Drake is the most overrated artist of all time and history will be brutal to him.",
    "Freddie Gibbs is the best pure rapper alive right now and it's not particularly close.",
    "Soundcloud rap was the worst thing to ever happen to the genre, no contest.",
    "Kanye hasn't made a classic since 2013 and the fans are coping hard about it.",
    "J Cole is mid and the platinum no features thing is the most overrated flex ever.",
    "Earl Sweatshirt fell off after Doris and the experimental stuff is just an excuse.",
    "Nicki Minaj is the greatest female rapper ever and the competition isn't close.",
    "Lil Uzi is more talented than half the lyrical miracle rappers people worship.",
    "Mach-Hommy is overpriced gatekept nonsense and the emperor has no clothes.",
    "21 Savage is a one trick pony and the monotone thing got old three projects ago.",
    "The 90s were the worst decade for rap production and nostalgia is lying to you.",
    "Pusha T is a top 5 lyricist of all time and Daytona proves it single handedly.",
    "Doja Cat is more talented than every rapper in the XXL freshman class combined.",
    "Jay-Z is overrated and rode the Roc-A-Fella machine more than his actual skill.",
    "Kid Cudi influenced more of today's sound than Kanye ever did and people forget that.",
    "Vince Staples is the smartest rapper alive and the sales prove the public can't handle it.",
    "Lil Baby is the voice of this generation whether the lyrical purists like it or not.",
    "Atlanta has produced more important rap than New York for fifteen years straight now.",
    "Tyler will never top Igor and he knows it, that's why he keeps switching styles.",
    "Joey Bada$$ is a nostalgia act with no original ideas and the cosign carried him.",
    "Kendrick is overrated and the Pulitzer was a political pick, the music doesn't back it.",
    "Drake single handedly made every rapper soft and the genre never recovered from it.",
    "Conscious rap is just preachy and boring and nobody actually listens to it twice.",
    "Roddy Ricch is a one album wonder and everyone saw it coming a mile away.",
    "Outkast is the best rap group ever and there is genuinely no real second place.",
    "Lil Yachty's Let's Start Here is the most overrated pivot album of the decade.",
    "Big Sean is secretly top tier and the hate is just a meme that went too far.",
    "Kanye is washed and the only reason people still care is the controversy not the music.",
    "Wu-Tang is overrated as a collective, only three of them could actually rap.",
    "Future invented modern rap and gets a fraction of the credit Drake stole from him.",
    "Cole's discography is the most consistent in rap and the slander is just contrarian.",
    "Megan is the most talented rapper of the new wave and it isn't remotely close.",
    "Logic is unironically a good rapper and the hate train ruined his career unfairly.",
    "Travis Scott hasn't rapped a memorable verse in his life and Utopia proved it.",
    "DOOM is the most influential underground rapper ever and mainstream heads will never get it.",
    "Kendrick vs Drake wasn't even close, Drake lost the second the first track dropped.",
    "Gunna is more talented than Young Thug and the loyalty crowd can't handle that truth.",
    "The best rap decade was the 2010s and the boom bap purists are simply wrong.",
    "Aesop Rock is unlistenable and the vocabulary thing is a gimmick not a skill.",
    "Tyler is the only rapper from his era who actually grew, the rest stayed the same.",
    "Lil Wayne ruined a generation of rappers by making mumble punchlines cool.",
    "Kanye's worst album still clears Drake's entire discography and it's not debatable.",
    "Nas is a top 2 rapper ever and anyone who says otherwise is just being edgy.",
    "Doechii is the most overrated artist of the year and the Grammy was premature.",
    "Eminem hasn't made a good song since 2010 and the fanbase is in total denial.",
]

# ----------------------------------------------------------------------------
# STAN  - emotional fan reaction; feeling is the point, little/no argument
# ----------------------------------------------------------------------------
stan = [
    "ALBUM OF THE YEAR no skips no notes I've been screaming since midnight",
    "bro I am not okay after that last track I had to sit in my car for ten minutes",
    "WE ARE SO BACK the king has returned I'm literally shaking right now",
    "been waiting four years for this drop and it did NOT disappoint oh my god",
    "track 7 just changed my entire life I can't stop crying in the gym",
    "this man can do no wrong I would die for him genuinely",
    "the snippet alone has me FERAL I cannot wait for the full thing aaaaa",
    "playing this on repeat until I dissolve into the floor goodbye everyone",
    "I've listened to this 14 times today and I have work in the morning send help",
    "MY GOAT DROPPED nothing else matters today I'm calling out of work",
    "the way I gasped when the beat switched I woke up my whole apartment",
    "no because why am I crying at the gym to a rap album this is embarrassing",
    "instant classic I don't even need to think about it I just KNOW",
    "this is the best thing he's ever done I will not be taking questions",
    "I have goosebumps the entire way through I genuinely cannot function",
    "the wait was so worth it I'm never doubting the rollout again I promise",
    "bro really woke up and decided to end careers today absolute massacre",
    "I'm gonna be playing this all summer no other album exists to me now",
    "literally vibrating right now this is everything I wanted and more",
    "the visuals dropped too?? I'm having a full breakdown this is too much",
    "ten years from now I'll remember exactly where I was when this dropped",
    "okay I'm sobbing the outro got me good no warning at all just devastation",
    "this is the soundtrack of my entire summer I already know it",
    "screaming crying throwing up the rollout for this was PERFECT",
    "I can't believe we get to witness him in real time we are so blessed",
    "the bars the beats the everything I'm not surviving this week",
    "first listen reaction: speechless. second listen: still speechless. legend.",
    "I would give up my left arm for a tour announcement right now please",
    "this dropped at 3am my time and I have NO regrets about staying up",
    "best in the game it's not even a competition I'm just grateful he exists",
    "the features ATE every single one of them came to play oh my goodness",
    "I'm putting this on every playlist I own immediately no skips fr fr",
    "this is the album I'm gonna show my kids one day mark my words",
    "the man is simply different we don't deserve him honestly",
    "okay the deluxe better come soon because I already need MORE",
    "running this back for the fifth time and it keeps getting better help",
    "I knew it was gonna be good but I did NOT know it was gonna be THIS good",
    "the energy on this is unreal I feel like I can run through a wall",
    "him snapping on a Tuesday for no reason we are blessed beyond measure",
    "no skips. NO SKIPS. do you understand what I'm telling you right now",
    "I haven't been this excited for a drop since high school what a feeling",
    "the cover art alone had me hype and then the music delivered too somehow",
    "this whole project is a vibe I've had it on since I woke up and won't stop",
    "the goat returns and all is right in the world again finally",
    "I'm emotionally compromised after track 9 do not perceive me today",
    "literally the only thing getting me through this week is this album",
    "the wait nearly killed me but I would do it all again for this",
    "I called it I CALLED IT this is his magnum opus and you can't tell me different",
    "I'm not even a big fan normally but this one got me good ngl",
    "we eating GOOD this year the drops have been insane and this tops them all",
    "the way the whole sub is losing it right now we are all so back together",
    "this is the kind of album that makes you remember why you love rap",
    "midnight release and I'm wide awake buzzing this is everything",
    "him really said let me end the year with a classic and just DID it",
    "every single track is a single I can't pick a favorite it's impossible",
    "I've been smiling like an idiot for forty minutes straight thank you sir",
    "the anticipation almost broke me but the payoff was worth every second",
    "I'm gonna need a week to recover from this one genuinely shook to my core",
    "this man understood the assignment and then rewrote the assignment entirely",
    "the replay value is insane I'm on loop three and not bored once",
    "okay this is officially my personality for the next six months sorry not sorry",
    "we waited and he DELIVERED I have never been so happy to be wrong about a delay",
    "the bridge on track 4 made me pull over while driving I am not okay",
    "I'm telling everyone I know to drop everything and listen RIGHT NOW",
    "best rollout best album best year I'm so glad I'm alive to hear this",
    "I have chills from start to finish and I refuse to take it off repeat",
    "this is the one. THIS is the one. I've never been more sure of anything.",
    "the whole car sang every word last night this album already feels like home",
    "I genuinely teared up at the closer no album has done that to me in years",
    "him dropping with zero warning and bodying everyone is so iconic of him",
]

# ----------------------------------------------------------------------------
# Difficult / borderline cases  (carry a note explaining the decision)
# ----------------------------------------------------------------------------
borderline = [
    ("This verse gave me literal chills, the way he flips from triplets into a straight flow right at the bridge is insane.",
     "critique",
     "Emotional opener (chills) reads stan, but it names a specific identifiable musical element (the triplet->straight flow flip at the bridge) as the REASON. Rule: if a specific element is cited as the cause of the feeling -> critique."),
    ("He's got more number one albums than anyone this decade so the GOAT debate is honestly over.",
     "hot_take",
     "Cites a real chart fact, which looks like critique. But the stat is decorative/cherry-picked to win a ranking argument, not analysis of the music. Rule: stat used to crown a verdict, not to reason about craft -> hot_take."),
    ("Album of the year, track 4 track 7 and track 11 are all bangers, literally no skips on this thing.",
     "stan",
     "Names specific tracks, which looks like critique. But there is no argument about WHY they work, just enthusiastic listing. Rule: naming tracks without evaluating the music -> stan."),
    ("Tyler is the best artist of his generation and Igor's chord progressions prove he's operating on another level.",
     "hot_take",
     "Gestures at chords (critique-flavored) but the chord claim is unsupported and bolted onto a GOAT verdict. The verdict is the point, the 'evidence' is a name-drop. -> hot_take."),
    ("I cannot believe how clean the mix is on this, every instrument has its own space and nothing fights for room.",
     "critique",
     "High enthusiasm could read stan, but the post makes a specific, verifiable claim about the mix (instrument separation / no masking). Reasoning survives removing the excitement. -> critique."),
]

# ----------------------------------------------------------------------------
# Assemble + write
# ----------------------------------------------------------------------------
def main():
    rows = []
    for t in critique:
        rows.append((t, "critique", ""))
    for t in hot_take:
        rows.append((t, "hot_take", ""))
    for t in stan:
        rows.append((t, "stan", ""))
    for t, lab, note in borderline:
        rows.append((t, lab, note))

    counts = {}
    for _, lab, _ in rows:
        counts[lab] = counts.get(lab, 0) + 1

    with open("takemeter_dataset.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "label", "notes"])
        w.writerows(rows)

    print(f"Wrote takemeter_dataset.csv with {len(rows)} examples")
    for lab in sorted(counts):
        print(f"  {lab:10s} {counts[lab]:3d}  ({counts[lab]/len(rows)*100:.1f}%)")


if __name__ == "__main__":
    main()
