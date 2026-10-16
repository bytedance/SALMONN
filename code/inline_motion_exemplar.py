"""Shared action-family matching and inline-exemplar utilities.

The matcher is deliberately lexical and high precision.  Training-data
construction and streaming inference import this same file so that an action
is triggered by exactly the same rules in both paths.
"""

from __future__ import annotations

import re

import torch

from utils import rotation_conversions as rc


# Reuse two existing single-token Qwen delimiters.  This avoids resizing the
# frozen/checkpoint vocabulary and leaves the no-exemplar path byte-identical.
EXEMPLAR_START_TOKEN = "<|object_ref_start|>"
EXEMPLAR_END_TOKEN = "<|object_ref_end|>"


ACTION_FAMILIES = {
    "phone_call": {
        "triggers": [r"\bphone call\b", r"\bcall me\b", r"\bon the phone\b", r"\bget on (?:my|the|your|his|her|a) phone\b", r"\btelephone\b", r"\bface ?time\b", r"\banswer(?:ed|ing)? (?:my|the|your|his|her|a) phone\b", r"\bkeep in touch\b"],
        "labels": [r"\bcall me\b.*(?:sign|gesture)", r"\b(?:phone|telephone)\b.*\b(?:ear|call)\b", r"\b(?:ear|call)\b.*\b(?:phone|telephone)\b", r"mimes? (?:making|answering|taking).*\bcall\b", r"\bface ?time\b.*(?:mime|phone|gesture)"],
    },
    "phone_texting": {
        "triggers": [r"\btext(?:ed|ing|s)?\b", r"\bsmartphone\b", r"\bcell ?phone\b"],
        "labels": [r"(?:phone|smartphone).*(?:text|typ|thumb|scroll)", r"(?:text|typ|thumb|scroll).*(?:phone|smartphone)"],
    },
    "drink": {
        "triggers": [r"\bdrink(?:ing|s|ed)?\b", r"\bsip(?:ping|ped|s)?\b", r"\bcup of\b", r"\bglass of\b"],
        "labels": [r"mimes?.*(?:drink|sip)", r"(?:drink|sip).*(?:cup|glass|mouth)", r"(?:cup|glass).*(?:mouth|drink|sip)"],
    },
    "eat": {
        "triggers": [r"\beat(?:ing|en|s)?\b", r"\bbite(?:s|n)?\b", r"\bchew(?:ed|ing|s)?\b"],
        "labels": [r"mimes?.*\b(?:eat(?:ing)?|bite)\b", r"\b(?:food|bite)\b.*\b(?:mouth|eat(?:ing)?)\b", r"(?:hand|finger).*mouth.*\b(?:eat(?:ing)?|food)\b"],
    },
    "write_draw": {
        "triggers": [r"\bwrit(?:e|es|ing|ten)\b", r"\bdraw(?:ing|s|n)?\b", r"\bpen(?:cil)?\b"],
        "labels": [r"mimes?.*(?:writ|draw)", r"(?:writ|draw).*(?:pen|paper|surface|hand)", r"(?:pen|pencil).*(?:writ|draw)"],
    },
    "drive_steer": {
        "triggers": [r"\bdriv(?:e|es|ing|en)\b", r"\bsteer(?:ing|s|ed)?\b", r"\bsteering wheel\b"],
        "labels": [r"(?:steering wheel|driv(?:e|ing))", r"mimes?.*(?:steer|driv)"],
    },
    "lift_weights": {
        "triggers": [r"\bweight(?:s|lifting)?\b", r"\bdumbbell(?:s)?\b", r"\bbicep curl(?:s)?\b", r"\blift(?:ing)? weights\b"],
        "labels": [r"(?:lift|curl|holding).*(?:weight|dumbbell)", r"(?:weight|dumbbell).*(?:lift|curl|holding)"],
    },
    "clap_applause": {
        "triggers": [r"\bclap(?:ped|ping|s)?\b", r"\bapplau(?:se|ded|ding)\b"],
        "labels": [r"\bclap(?:s|ped|ping)?\b", r"\bapplau(?:se|ding|ded)\b"],
    },
    "stop_hold": {
        "triggers": [r"\bstop(?:ped|ping|s)?\b", r"\bhold on\b"],
        "labels": [r"(?:stop|hold on|wait).*(?:gesture|palm|hand)", r"(?:hand|palm).*(?:stop|hold on|wait)"],
    },
    "heart_love": {
        "triggers": [r"\bheart\b", r"\bvalentine(?:'s|s)?\b"],
        "labels": [r"heart shape", r"(?:heart|love).*(?:hands|fingers|gesture)", r"(?:hands|fingers).*(?:heart|love)"],
    },
    "small_amount": {
        "triggers": [r"\btiny\b", r"\bsmall amount\b", r"\bthis small\b"],
        "labels": [r"(?:pinch|pinches|pinching).*(?:small|little|tiny|amount)", r"(?:small|little|tiny|amount).*(?:pinch|finger|thumb)"],
    },
    "large_size": {
        "triggers": [r"\bhuge\b", r"\benormous\b", r"\bvery big\b", r"\bthis big\b", r"\breally big\b", r"\bbigger\b", r"\blarge\b"],
        "labels": [r"(?:arms|hands).*(?:wide|apart|spread).*(?:size|large|big|huge|enormous)", r"(?:size|large|big|huge|enormous).*(?:arms|hands).*(?:wide|apart|spread)", r"(?:illustrate|show|represent).*(?:size|large|big|huge|enormous).*(?:arms|hands|spread)"],
    },
    "together_join": {
        "triggers": [r"\btogether\b", r"\bjoin(?:ed|ing|s)?\b", r"\bcombin(?:e|ed|ing|es)\b", r"\bconnect(?:ed|ing|s)?\b"],
        "labels": [r"(?:hands|fingers).*(?:together|join|combin|connect|interlac)", r"(?:together|join|combin|connect|interlac).*(?:hands|fingers)"],
    },
    "thinking_temple": {
        "triggers": [r"\bbrain\b", r"\bcrazy\b", r"\blost my mind\b", r"\bout of (?:my|your|his|her|their) mind\b"],
        "labels": [r"(?:tap|touch|point).*(?:temple|head).*(?:think|brain|mind|crazy)", r"(?:think|brain|mind|crazy).*(?:temple|head)"],
    },
    "crying": {
        "triggers": [r"\bcry(?:ing|ied|ies)?\b", r"\btear(?:s|ful)?\b", r"\bsobb(?:ing|ed)?\b"],
        "labels": [r"mimes?.*(?:cry|tear|sob)", r"(?:cry|tear|sob).*(?:eye|face|fist|hand)"],
    },
    "pray_thanks": {
        "triggers": [r"\bpray(?:ing|ed|s|er)?\b", r"\bthank(?:s|ed|ing|ful)?\b", r"\bgrateful\b"],
        "labels": [r"(?:palms|hands).*(?:pray|prayer|thank|grateful)", r"(?:pray|prayer|thank|grateful).*(?:palms|hands|gesture)"],
    },
    "brush_teeth": {
        "triggers": [r"\bbrush(?:ed|ing|es)? (?:my|your|his|her|the)? ?teeth\b", r"\btoothbrush\b"],
        "labels": [r"(?:brush|toothbrush).*(?:teeth|mouth)", r"(?:teeth|mouth).*(?:brush|toothbrush)"],
    },
    "guitar": {
        "triggers": [r"\bguitar\b", r"\bstrum(?:med|ming|s)?\b"],
        "labels": [r"(?:guitar|strum).*(?:mime|hand|play)", r"(?:mime|hand|play).*(?:guitar|strum)"],
    },
    "pour": {
        "triggers": [r"\bpour(?:ed|ing|s)?\b"],
        "labels": [r"mimes?.*pour", r"pour.*(?:bottle|cup|glass|hand|liquid)"],
    },
    "backchannel_nod": {
        "triggers": [r"\bmm ?hmm\b", r"\bmhm+\b", r"\buh ?huh\b", r"\bhmm+\b", r"\byeah\b", r"\byep\b"],
        "labels": [r"\b(?:head )?nod(?:s|ded|ding)?\b", r"\bhead bob(?:s|bed|bing)?\b"],
    },
}

# Oracle-only families are allowed here as well.  The data-index builder uses
# motion descriptions to expose every supported family during training, then
# separately marks the high-precision transcript-triggered inference subset.
ACTION_FAMILIES.update({
    "point_self": {
        "triggers": [r"\bmyself\b", r"\bme personally\b", r"\bme\b"],
        "labels": [r"(?:point|thumb).*(?:own chest|himself|herself|themself|myself)", r"(?:own chest|himself|herself).*(?:point|thumb)"],
    },
    "point_other": {
        "triggers": [r"\byou personally\b", r"\byourself\b", r"\byou\b"],
        "labels": [r"(?:point|index finger).*(?:other person|interlocutor|conversational partner|listener)", r"(?:other person|interlocutor|conversational partner|listener).*(?:point|index finger)"],
    },
    "point_location": {
        "triggers": [r"\bover there\b", r"\bright here\b", r"\bright there\b", r"\bthere\b", r"\bhere\b"],
        "labels": [r"(?:point|deictic).*(?:location|place|spot|area|here|there)", r"(?:location|place|spot|area).*(?:point|deictic)"],
    },
    "point_behind": {
        "triggers": [r"\bbehind\b", r"\bback there\b", r"\bover (?:my|your|his|her) shoulder\b"],
        "labels": [r"(?:thumb|point).*(?:over (?:his|her|their) shoulder|behind|backward)", r"(?:behind|backward).*(?:thumb|point)"],
    },
    "point_up": {
        "triggers": [r"\bup there\b", r"\babove\b", r"\bupstairs\b", r"\bsky\b", r"\bceiling\b"],
        "labels": [r"(?:point|index finger).*(?:upward|above|sky|ceiling|up high)", r"(?:upward|above|sky|ceiling).*(?:point|index finger)"],
    },
    "point_down": {
        "triggers": [r"\bdown there\b", r"\bbelow\b", r"\bdownstairs\b", r"\bground\b", r"\bfloor\b"],
        "labels": [r"(?:point|index finger).*(?:downward|below|ground|floor)", r"(?:downward|below|ground|floor).*(?:point|index finger)"],
    },
    "count_one": {
        "triggers": [r"\bnumber one\b", r"\bone (?:thing|time|person|way|reason|point)\b", r"\bfirst\b"],
        "labels": [r"(?:one|single).*(?:finger|number|count)", r"(?:finger|number|count).*\bone\b"],
    },
    "count_two": {
        "triggers": [r"\bnumber two\b", r"\btwo (?:things|times|people|ways|reasons|points)\b", r"\bsecond\b"],
        "labels": [r"\btwo\b.*(?:finger|number|count)", r"(?:finger|number|count).*\btwo\b"],
    },
    "count_three": {
        "triggers": [r"\bnumber three\b", r"\bthree\b", r"\bthird\b"],
        "labels": [r"\bthree\b.*(?:finger|number|count)", r"(?:finger|number|count).*\bthree\b"],
    },
    "count_four": {
        "triggers": [r"\bnumber four\b", r"\bfour\b", r"\bfourth\b"],
        "labels": [r"\bfour\b.*(?:finger|number|count)", r"(?:finger|number|count).*\bfour\b"],
    },
    "count_five": {
        "triggers": [r"\bnumber five\b", r"\bfive\b", r"\bfifth\b"],
        "labels": [r"\bfive\b.*(?:finger|number|count)", r"(?:finger|number|count).*\bfive\b"],
    },
    "thumbs_up": {
        "triggers": [r"\bthumbs up\b", r"\bgood job\b", r"\bwell done\b"],
        "labels": [r"thumbs?[- ]up", r"(?:raised|extends?|holds?).*thumb.*(?:approval|positive|good)"],
    },
    "air_quotes": {
        "triggers": [r"\bquote unquote\b", r"\bso called\b", r"\bair quotes\b"],
        "labels": [r"air[- ]quotes?", r"quotation marks?.*(?:finger|hand)", r"(?:finger|hand).*quotation marks?"],
    },
    "shrug_uncertain": {
        "triggers": [r"\bi don'?t know\b", r"\bwho knows\b", r"\bnot sure\b", r"\bmaybe\b"],
        "labels": [r"\bshrug(?:s|ged|ging)?\b", r"(?:palms|shoulders).*(?:uncertain|don'?t know|helpless|questioning)"],
    },
    "head_shake_no": {
        "triggers": [r"\bno\b", r"\bnot at all\b", r"\bnever\b"],
        "labels": [r"shakes? (?:his |her |their )?head.*(?:no|negat|disagree)", r"(?:no|negat|disagree).*shakes? (?:his |her |their )?head"],
    },
    "finger_wag_no": {
        "triggers": [r"\bno no\b", r"\bdon'?t do that\b", r"\bnot allowed\b"],
        "labels": [r"(?:wag|wags|wagging).*(?:index )?finger", r"(?:index )?finger.*(?:side.to.side|negation|admonish)"],
    },
    "wave_greeting": {
        "triggers": [r"\bhello\b", r"\bhi there\b", r"\bgoodbye\b", r"\bbye(?: bye)?\b"],
        "labels": [r"(?:wave|waves|waving).*(?:hello|greet|goodbye|bye)", r"(?:hello|greet|goodbye|bye).*(?:wave|waving)"],
    },
    "dismiss_wave": {
        "triggers": [r"\bwhatever\b", r"\bgo away\b", r"\bget rid of\b", r"\bforget it\b"],
        "labels": [r"dismissive (?:wave|motion)", r"(?:wave|sweep|brush).*(?:away|dismiss|discard|reject)"],
    },
    "open_palms_question": {
        "triggers": [r"\bwhat (?:is|are|about|do|can|should|would)\b", r"\bhow (?:is|are|do|can|should|would)\b", r"\bwhy\b"],
        "labels": [r"(?:open|raises?|holds?).*palms?.*(?:question|asking|what|how|why)", r"(?:question|asking|what|how|why).*(?:open )?palms?"],
    },
    "hand_on_heart": {
        "triggers": [r"\bfrom the heart\b", r"\bmy heart\b", r"\bi promise\b", r"\bhonestly\b"],
        "labels": [r"(?:hand|palm).*(?:over|on|to).*(?:heart|chest).*(?:emotion|sincere|promise|heart)", r"(?:heart|sincere|promise).*(?:hand|palm).*(?:chest|heart)"],
    },
    "arms_crossed": {
        "triggers": [r"\bcross(?:ed|ing)? (?:my|your|his|her|the) arms\b", r"\bfold(?:ed|ing)? (?:my|your|his|her|the) arms\b"],
        "labels": [r"(?:cross|fold).*(?:arms|hands).*(?:chest|body)", r"(?:arms|hands).*(?:cross|fold).*(?:chest|body)"],
    },
    "hug_embrace": {
        "triggers": [r"\bhug(?:ged|ging|s)?\b", r"\bembrac(?:e|ed|ing)\b"],
        "labels": [r"mimes?.*(?:hug|embrace)", r"(?:arms|hands).*(?:hug|embrace|around someone)"],
    },
    "handshake": {
        "triggers": [r"\bhandshake\b", r"\bshake hands\b", r"\bshook hands\b"],
        "labels": [r"(?:mime|imitat|gesture).*(?:handshake|shak(?:e|ing) hands)", r"handshake.*(?:hand|motion)"],
    },
    "high_five": {
        "triggers": [r"\bhigh five\b"],
        "labels": [r"high[- ]five", r"raised palm.*(?:celebrat|slap)"],
    },
    "fingers_crossed": {
        "triggers": [r"\bfingers crossed\b", r"\bgood luck\b", r"\bhope so\b"],
        "labels": [r"cross(?:es|ed|ing)? (?:his |her |their )?fingers", r"fingers.*crossed.*(?:hope|luck)"],
    },
    "strong_flex": {
        "triggers": [r"\bstrong\b", r"\bstrength\b", r"\bmuscle(?:s)?\b", r"\bpowerful\b"],
        "labels": [r"(?:flex|bends?).*(?:bicep|muscle|strong|strength)", r"(?:strong|strength|muscle).*(?:flex|fist|bicep)"],
    },
    "throw": {
        "triggers": [r"\bthrow(?:s|ing|n)?\b", r"\bthrew\b", r"\btoss(?:ed|ing|es)?\b"],
        "labels": [r"mimes?.*(?:throw|toss)", r"(?:throw|toss).*(?:arm|hand|object|motion)"],
    },
    "catch": {
        "triggers": [r"\bcatch(?:es|ing)?\b", r"\bcaught\b"],
        "labels": [r"mimes?.*(?:catch|caught)", r"(?:catch|caught).*(?:ball|object|hands|motion)"],
    },
    "push_away": {
        "triggers": [r"\bpush(?:ed|ing|es)? (?:it |them |him |her )?away\b", r"\bkeep (?:it|them|him|her) away\b"],
        "labels": [r"\b(?:push(?:es|ed|ing)?|press(?:es|ed|ing)?|shove(?:s|d|ing)?)\b.*\b(?:away|outward|forward)\b", r"\b(?:away|outward)\b.*\b(?:push(?:es|ed|ing)?|press(?:es|ed|ing)?|shove(?:s|d|ing)?)\b"],
    },
    "pull_toward": {
        "triggers": [r"\bpull(?:ed|ing|s)?\b", r"\bbring (?:it|them|him|her) (?:in|closer)\b"],
        "labels": [r"(?:pull|draw).*(?:toward|inward|closer|back)", r"(?:toward|inward|closer).*(?:pull|draw)"],
    },
    "give_offer": {
        "triggers": [r"\bgive (?:me|you|him|her|them|it)\b", r"\bgave (?:me|you|him|her|them|it)\b", r"\boffer(?:ed|ing|s)?\b"],
        "labels": [r"(?:mime|hand|arms?).*(?:giv|offer|present).*(?:object|something|item)", r"(?:giv|offer|present).*(?:outstretched|forward|hand|object)"],
    },
    "take_pickup": {
        "triggers": [r"\bpick(?:ed|ing)? (?:it |them |him |her )?up\b", r"\btake (?:it|them|this|that)\b", r"\bgrab(?:bed|bing|s)?\b"],
        "labels": [r"(?:mime|hand).*(?:pick(?:ing)? up|take|grab).*(?:object|item|something)", r"(?:pick(?:ing)? up|take|grab).*(?:object|item|hand)"],
    },
    "open_door": {
        "triggers": [r"\bopen(?:ed|ing)? (?:the |a |my |your )?(?:door|window|curtain|lid)\b"],
        "labels": [r"mimes?.*open(?:ing)?.*(?:door|window|curtain|lid)", r"(?:door|window|curtain|lid).*(?:open|pull|slide)"],
    },
    "close_door": {
        "triggers": [r"\bclos(?:e|ed|ing) (?:the |a |my |your )?(?:door|window|curtain|lid)\b", r"\bshut (?:the |a )?(?:door|window|lid)\b"],
        "labels": [r"mimes?.*(?:clos|shut).*(?:door|window|curtain|lid)", r"(?:door|window|curtain|lid).*(?:clos|shut)"],
    },
    "type_keyboard": {
        "triggers": [r"\btyp(?:e|ed|ing)\b", r"\bkeyboard\b", r"\bcomputer\b"],
        "labels": [r"(?:mime|imitat).*(?:typ|keyboard)", r"(?:typ|keyboard).*(?:finger|hand|computer)"],
    },
    "read_book": {
        "triggers": [r"\bread(?:ing)? (?:a |the |this )?(?:book|letter|paper)\b", r"\bbook\b"],
        "labels": [r"mimes?.*\b(?:read(?:ing)?|book)\b", r"\b(?:read(?:ing)?|book|pages)\b.*(?:hold|open|turn|hand)"],
    },
    "camera_photo": {
        "triggers": [r"\btake (?:a |the )?(?:photo|picture)\b", r"\bcamera\b", r"\bphotograph(?:ed|ing|s)?\b"],
        "labels": [r"mimes?.*(?:camera|photo|picture)", r"(?:camera|photo|picture).*(?:hold|frame|button|hand)"],
    },
    "cook_stir": {
        "triggers": [r"\bstir(?:red|ring|s)?\b", r"\bmix(?:ed|ing|es)?\b", r"\bcook(?:ed|ing|s)?\b"],
        "labels": [r"mimes?.*(?:stir|mix|cook)", r"(?:stir|mix|cook).*(?:pot|bowl|spoon|hand|circular)"],
    },
    "chop_cut": {
        "triggers": [r"\bchop(?:ped|ping|s)?\b", r"\bcut(?:s|ting)?\b", r"\bknife\b"],
        "labels": [r"mimes?.*(?:chop|cut)", r"(?:chop|cut|knife).*(?:food|hand|downward|surface)"],
    },
    "dance": {
        "triggers": [r"\bdanc(?:e|ed|ing|es)\b"],
        "labels": [r"(?:mime|imitat|body|arms?).*danc", r"danc.*(?:motion|movement|arms?|body)"],
    },
    "walk": {
        "triggers": [r"\bwalk(?:ed|ing|s)?\b", r"\bstroll(?:ed|ing|s)?\b"],
        "labels": [r"mimes?.*(?:walk|stroll)", r"(?:walk|stroll).*(?:finger|hand|motion|legs?)"],
    },
    "run_jog": {
        "triggers": [r"\brun(?:s|ning)?\b", r"\bran\b", r"\bjog(?:ged|ging|s)?\b"],
        "labels": [r"mimes?.*(?:run|jog)", r"(?:run|jog).*(?:arm|hand|motion|legs?)"],
    },
    "swim": {
        "triggers": [r"\bswim(?:s|ming)?\b", r"\bswam\b", r"\bpool\b"],
        "labels": [r"mimes?.*(?:swim|stroke)", r"(?:swim|stroke).*(?:arm|water|motion)"],
    },
    "jump": {
        "triggers": [r"\bjump(?:ed|ing|s)?\b", r"\bleap(?:ed|ing|s)?\b"],
        "labels": [r"mimes?.*(?:jump|leap)", r"(?:jump|leap).*(?:upward|body|motion|hands?)"],
    },
    "fly": {
        "triggers": [r"\bfly(?:ing|s)?\b", r"\bflew\b", r"\bflight\b"],
        "labels": [r"mimes?.*(?:fly|flight)", r"(?:fly|flight|bird).*(?:wing|arms?|motion)"],
    },
    "play_ball": {
        "triggers": [r"\bball\b", r"\bbasketball\b", r"\bfootball\b", r"\bbaseball\b"],
        "labels": [r"mimes?.*(?:ball|basketball|football|baseball)", r"(?:ball|basketball|football|baseball).*(?:throw|catch|dribble|swing|hand)"],
    },
    "punch_fight": {
        "triggers": [r"\bpunch(?:ed|ing|es)?\b", r"\bfight(?:ing|s)?\b", r"\bbox(?:ed|ing)\b"],
        "labels": [r"mimes?.*(?:punch|fight|boxing)", r"(?:punch|fight|boxing).*(?:fist|arm|hand)"],
    },
    "shape_circle": {
        "triggers": [r"\bcircle\b", r"\bcircular\b", r"\bround\b"],
        "labels": [r"(?:draw|trace|form|shape).*(?:circle|circular|round)", r"(?:circle|circular|round).*(?:draw|trace|form|shape|hand)"],
    },
    "shape_triangle": {
        "triggers": [r"\btriangle\b", r"\btriangular\b"],
        "labels": [r"(?:form|trace|shape).*(?:triangle|triangular)", r"(?:triangle|triangular).*(?:form|trace|shape|hand)"],
    },
    "flat_level": {
        "triggers": [r"\bflat\b", r"\blevel\b", r"\beven surface\b"],
        "labels": [r"(?:flat|horizontal|level).*(?:palm|hand|surface)", r"(?:palm|hand).*(?:flat|horizontal|level|even surface)"],
    },
    "tall_height": {
        "triggers": [r"\btall(?:er|est)?\b", r"\bhigh up\b", r"\bheight\b"],
        "labels": [r"\b(?:tall|height)\b.*(?:raise|high|vertical|hand)", r"(?:raise|holds?).*(?:hand|palm).*\b(?:tall|height|high)\b"],
    },
    "short_low": {
        "triggers": [r"\bshort(?:er|est)?\b", r"\blow down\b", r"\bnot tall\b"],
        "labels": [r"(?:short|low height).*(?:lower|low|hand)", r"(?:lower|holds?).*(?:hand|palm).*(?:short|low height)"],
    },
    "far_distance": {
        "triggers": [r"\bfar away\b", r"\blong distance\b", r"\bdistant\b", r"\bfar apart\b"],
        "labels": [r"(?:far|distance|distant).*(?:apart|wide|away|hands|arms)", r"(?:hands|arms).*(?:far|distance|distant|far apart)"],
    },
    "close_distance": {
        "triggers": [r"\bclose together\b", r"\bnear(?:by|er)?\b", r"\bshort distance\b"],
        "labels": [r"\b(?:close|near)\b.*(?:together|hands|fingers|space)|\bshort distance\b.*(?:hands|space)", r"(?:hands|fingers).*\b(?:close|near)\b|(?:hands|fingers).*\bshort distance\b"],
    },
    "many_quantity": {
        "triggers": [r"\bso many\b", r"\ba lot of\b", r"\blots of\b", r"\bmany\b"],
        "labels": [r"(?:many|large quantity|a lot).*(?:hands|arms|wide|spread)", r"(?:hands|arms).*(?:many|large quantity|a lot)"],
    },
    "all_whole": {
        "triggers": [r"\bthe whole\b", r"\beverything\b", r"\ball of (?:it|them|us)\b", r"\bentire\b"],
        "labels": [r"\b(?:all|whole|entire|everything)\b.*(?:sweep|circle|collect|gather|encompass|arms? wide)", r"(?:sweep|circle|collect|gather|encompass|arms? wide).*\b(?:all|whole|entire|everything)\b"],
    },
    "separate_apart": {
        "triggers": [r"\bseparate(?:d|ly)?\b", r"\bapart\b", r"\bdifferent sides\b", r"\bsplit(?:ting)?\b"],
        "labels": [r"(?:separate|apart|split).*(?:hands|arms|sides|outward)", r"(?:hands|arms).*(?:separate|apart|split)"],
    },
    "forward_progress": {
        "triggers": [r"\bmove forward\b", r"\bstep forward\b", r"\bprogress\b", r"\bgo ahead\b"],
        "labels": [r"(?:forward|ahead|progress).*(?:hand|arm|push|motion)", r"(?:hand|arm).*(?:forward|ahead|progress)"],
    },
    "past_back": {
        "triggers": [r"\bin the past\b", r"\bback then\b", r"\bpreviously\b", r"\bbefore that\b"],
        "labels": [r"(?:past|back then|previous).*(?:behind|backward|shoulder|hand)", r"(?:behind|backward|shoulder).*(?:past|previous|back then)"],
    },
    "long_length": {
        "triggers": [r"\bvery long\b", r"\blong way\b", r"\blong (?:time|line|road|distance)\b"],
        "labels": [r"\b(?:long|length)\b.*(?:apart|horizontal|extend|hands|arms)", r"(?:hands|arms).*\b(?:long|length)\b"],
    },
    "narrow_width": {
        "triggers": [r"\bnarrow\b", r"\bthin\b", r"\bskinny\b"],
        "labels": [r"\b(?:narrow|thin|skinny)\b.*(?:close|gap|hands|fingers)", r"(?:hands|fingers).*\b(?:narrow|thin|skinny)\b"],
    },
    "road_path": {
        "triggers": [r"\broad\b", r"\bstreet\b", r"\bpath\b", r"\btrack\b"],
        "labels": [r"(?:road|street|path|track).*(?:trace|line|forward|hand)", r"(?:trace|draw).*(?:road|street|path|track)"],
    },
    "wall_barrier": {
        "triggers": [r"\bwall\b", r"\bbarrier\b", r"\bblock(?:ed|ing)?\b"],
        "labels": [r"(?:wall|barrier|block).*(?:flat|vertical|palm|hand)", r"(?:palm|hand).*(?:wall|barrier|block)"],
    },
    "money_pay": {
        "triggers": [r"\bmoney\b", r"\bcash\b", r"\bpay(?:ing|ment|s|ed)?\b", r"\bcoin(?:s)?\b"],
        "labels": [r"(?:money|cash|pay|coin).*(?:finger|hand|rub|count)", r"(?:rub|count).*(?:finger|hand).*(?:money|cash|coin)"],
    },
    "listen_ear": {
        "triggers": [r"\blisten(?:ed|ing|s)?\b", r"\bhear(?:d|ing|s)?\b", r"\bear\b"],
        "labels": [r"(?:hand|finger|cup).*\bear\b", r"\bear\b.*(?:hand|finger|cup)"],
    },
    "speak_mouth": {
        "triggers": [r"\bspeak(?:ing|s)?\b", r"\btalk(?:ed|ing|s)?\b", r"\bmouth\b"],
        "labels": [r"(?:hand|finger|point).*\bmouth\b", r"\bmouth\b.*(?:hand|finger|point|opening)", r"(?:yapping|talking) motion", r"mimes?.*(?:speaking|talking).*(?:mouth|words?)"],
    },
    "quiet_shh": {
        "triggers": [r"\bshh+\b", r"\bbe quiet\b", r"\bkeep quiet\b", r"\bsilence\b"],
        "labels": [r"(?:finger|index).*(?:lips|mouth).*(?:quiet|silence|shh)", r"(?:quiet|silence|shh).*(?:finger|lips|mouth)"],
    },
    "sleep": {
        "triggers": [r"\bsleep(?:ing|s)?\b", r"\bslept\b", r"\bbed\b", r"\bnap(?:ped|ping|s)?\b"],
        "labels": [r"mimes?.*(?:sleep|nap)", r"(?:sleep|nap).*(?:head|hands|pillow|cheek)"],
    },
    "wash_clean": {
        "triggers": [r"\bwash(?:ed|ing|es)?\b", r"\bclean(?:ed|ing|s)?\b", r"\bscrub(?:bed|bing|s)?\b"],
        "labels": [r"mimes?.*(?:wash|clean|scrub)", r"(?:wash|clean|scrub).*(?:hands|surface|motion)"],
    },
    "sew_knit": {
        "triggers": [r"\bsew(?:ed|ing|n)?\b", r"\bknit(?:ted|ting|s)?\b", r"\bstitch(?:ed|ing|es)?\b"],
        "labels": [r"mimes?.*(?:sew|knit|stitch)", r"(?:sew|knit|stitch).*(?:needle|thread|hand|finger)"],
    },
    "plant_garden": {
        "triggers": [r"\bplant(?:ed|ing|s)?\b", r"\bgarden(?:ed|ing)?\b", r"\bdig(?:ging|s)?\b"],
        "labels": [r"mimes?.*(?:plant|garden|dig)", r"(?:plant|garden|dig).*(?:soil|ground|hand)"],
    },
    "fish": {
        "triggers": [r"\bfish(?:ed|ing|es)?\b", r"\bfishing rod\b", r"\breel(?:ed|ing|s)?\b"],
        "labels": [r"mimes?.*(?:fish|cast|reel)", r"(?:fish|cast|reel).*(?:rod|line|hand|arm)"],
    },
    "drum": {
        "triggers": [r"\bdrum(?:med|ming|s)?\b"],
        "labels": [r"mimes?.*(?:drum|drumming)", r"(?:drum|drumming).*(?:hand|stick|beat)"],
    },
    "roll": {
        "triggers": [r"\broll(?:ed|ing|s)?\b"],
        "labels": [r"mimes?.*(?:roll|rolling)", r"(?:roll|rolling).*(?:hand|circular|object)"],
    },
    "carry_hold": {
        "triggers": [r"\bcarry(?:ing|ied|ies)?\b", r"\bholding (?:a|the|this|that)\b"],
        "labels": [r"mimes?.*(?:carry|hold).*(?:object|box|bag|item)", r"(?:carry|holding).*(?:object|box|bag|item).*(?:arms|hands)"],
    },
    "reach": {
        "triggers": [r"\breach(?:ed|ing|es)? (?:for|toward|towards|out)\b"],
        "labels": [r"(?:reach|extend).*(?:for|toward|out).*(?:hand|arm|object)", r"(?:hand|arm).*(?:reach|extend).*(?:for|toward|out)"],
    },
    "agree_nod": {
        "triggers": [r"\bi agree\b", r"\bthat'?s right\b", r"\bexactly\b", r"\byes\b"],
        "labels": [r"(?:nod|nods|nodding).*(?:agree|yes|affirm)", r"(?:agree|yes|affirm).*(?:nod|head)"],
    },
    "victory_celebrate": {
        "triggers": [r"\bvictory\b", r"\bwe won\b", r"\bi won\b", r"\bcelebrat(?:e|ed|ing)\b", r"\byay\b"],
        "labels": [r"(?:raise|pump|throw).*(?:fist|arms?).*(?:victory|celebrat|won|triumph)", r"(?:victory|celebrat|triumph).*(?:fist|arms?|hands?)"],
    },
    "balance_scale": {
        "triggers": [r"\bbalance\b", r"\bweigh(?:ed|ing|s)?\b", r"\bon the other hand\b"],
        "labels": [r"(?:balance|weigh|compare).*(?:hands|palms|sides|scale)", r"(?:hands|palms).*(?:balance|weigh|compare|scale)"],
    },
    "heavy_weight": {
        "triggers": [r"\bheavy\b", r"\bweighs a lot\b"],
        "labels": [r"(?:heavy|weight).*(?:strain|lower|lift|hands|arms)", r"(?:strain|lift).*(?:heavy|weight)"],
    },
    "light_weight": {
        "triggers": [r"\blightweight\b", r"\bvery light\b", r"\bweighs nothing\b"],
        "labels": [r"(?:light|lightweight|weightless).*(?:lift|float|hand|fingers)", r"(?:lift|float).*(?:light|lightweight|weightless)"],
    },
})


def normalize_text(text: str) -> str:
    text = str(text or "").lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def label_action_families(description: str):
    """Return high-precision pseudo-ground-truth families from motion prose."""
    text = normalize_text(description)
    return [
        family
        for family, rules in ACTION_FAMILIES.items()
        if any(re.search(pattern, text) for pattern in rules["labels"])
    ]


def match_action_families(text: str):
    """Return lexical trigger matches in deterministic taxonomy order."""
    text = normalize_text(text)
    matches = []
    for family, rules in ACTION_FAMILIES.items():
        spans = []
        for pattern in rules["triggers"]:
            spans.extend((m.start(), m.end(), m.group(0)) for m in re.finditer(pattern, text))
        if spans:
            spans.sort()
            matches.append({"action_family": family, "match": spans[0][2], "char_start": spans[0][0]})
    return sorted(matches, key=lambda row: (row["char_start"], row["action_family"]))


def exemplar_blend_weights(length: int, blend_frames: int, device=None):
    """C2 blend weights: 0->1, exact core, then 1->0."""
    if length <= 2 * blend_frames:
        raise ValueError("Exemplar length must be larger than two blend windows.")
    weights = torch.ones(int(length), dtype=torch.float32, device=device)
    phase = torch.linspace(0.0, 1.0, int(blend_frames), device=device)
    smooth = phase**3 * (phase * (phase * 6.0 - 15.0) + 10.0)
    weights[:blend_frames] = smooth
    weights[-blend_frames:] = torch.flip(smooth, dims=(0,))
    return weights


def slerp_axis_angle(target_axis_angle: torch.Tensor, reference_axis_angle: torch.Tensor, weights):
    """Geodesically blend matching (..., 3) axis-angle trajectories."""
    if target_axis_angle.shape != reference_axis_angle.shape:
        raise ValueError("Target and reference rotations must have identical shapes.")
    target_q = rc.axis_angle_to_quaternion(target_axis_angle)
    reference_q = rc.axis_angle_to_quaternion(reference_axis_angle)
    dot = (target_q * reference_q).sum(dim=-1, keepdim=True)
    reference_q = torch.where(dot < 0.0, -reference_q, reference_q)
    dot = (target_q * reference_q).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    weight_shape = [int(weights.shape[0])] + [1] * (target_q.ndim - 1)
    weight = weights.reshape(weight_shape).to(dtype=target_q.dtype, device=target_q.device)
    s0 = torch.sin((1.0 - weight) * theta) / sin_theta.clamp_min(1e-7)
    s1 = torch.sin(weight * theta) / sin_theta.clamp_min(1e-7)
    spherical = s0 * target_q + s1 * reference_q
    linear = torch.nn.functional.normalize((1.0 - weight) * target_q + weight * reference_q, dim=-1)
    blended_q = torch.where((sin_theta.abs() < 1e-5).expand_as(spherical), linear, spherical)
    return rc.quaternion_to_axis_angle(torch.nn.functional.normalize(blended_q, dim=-1))
