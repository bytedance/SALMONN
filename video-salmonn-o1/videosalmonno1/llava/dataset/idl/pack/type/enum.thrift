namespace go tiktok.pack_type.enum
namespace py tiktok.pack_type.enum

enum ShareStoryStatusEnum {
    UnknownShareStoryStatus = 0; // Unknown status
    DisableMentionTagShareStory = 1; // Has shared to story. Disable and do not show button
    EnableMentionTagShareStory = 2; // Has not shared to story. Enable and show button
}

enum AigcLabelType {
  AigcLabelTypeUnknown = 0;
  AigcLabelTypeManual = 1;
  AigcLabelTypeAuto = 2;
}

enum AnchorState {
    VISIBLE = 0
    INVISIBLE = 1
    SELF_VISIBLE = 2
}

enum AnchorType {
    WIKIPEDIA = 0   // wikipeida
    SHOP_LINK = 6   // MT shop link
    YELP = 8        // MT yelp anchor
    TRIPADVISOR = 9 // MT TripAdvisor anchor
    GAME = 10
    MEDIUM = 11 //Film and television variety show
    MT_MOVIE = 18
    DONATION = 19
    RESSO = 23
    TT_EFFECT = 28;
    TT_TEMPLATE = 29;
    THIRD_PARTY_GENERAL = 44;
    TT_EDIT_EFFECT = 64;
    TICKETMASTER = 69;
}
enum FriendsStatusEnum {
    UNKNOWN = 0        #default value
    FOLLOW_FRIENDS = 1  #following freind status
    MUTUAL_FRIENDS = 2  #mutual following friend status
    MATCHED_FRIENDS = 3 #Not following friend status
}
enum AnchorGeneralType {
    GENERAL_ANCHOR = 1
}

enum AnchorActionType {
    SHOOT = 1
}

enum AwemeDetailRequestSourceTypeEnum {
    NONE = 0, // no source
    COMMENT = 1,
    LONGVIDEO = 2, // from long video recommend tab
    IM = 3, // from im
}

enum CountStatusEnum {
    SUCCESS = 0
    FAIL = 1
}

enum PackLevelEnum {
    NONE = 0
    LITE = 1    # include object itself
    NORMAL = 2  # include related outreach objects
    FULL = 3    # include all outreach objects
}

enum PackSourceEnum {
    NONE = 0
    CHALLENGE_DETAIL = 1
    CHALLENGE_AWEME = 2
    COMMENT_LIST = 3
    FAVORITE = 4
    FEED = 5
    HOTSEARCH_BILLBOARD = 6
    MUSIC_DETAIL = 7
    MUSIC_AWEME = 8
    PROFILE = 9
    PUBLISH = 10
    SEARCH = 11
    FANS_CLUB = 12
    FOLLOW_FEED = 13
    STORY = 14
    LIFE = 15
    CATEGORY_LIST = 16
    STICKER_DETAIL = 17
    STICKER_AWEME = 18
    MP_PLATFORM = 19
    MUSIC_LIST = 20
    SQUARE_FEED = 21
    WIDGET_FEED = 22
    CARD_LIST = 23
    IM_UNREAD_LIST = 24
    MICRO_APP_LIST = 25
    AWEME_DETAIL = 26
    MUSIC_RECOMMEND_LIST = 27
    POST = 28
    Deprecated29 = 29
    Deprecated30 = 30
    BRAND_BILLBOARD = 31
    ECOMMERCE_FEED = 32
    ECOMMERCE_BOLT = 33
    MAGIC_ACTIVITY_AWEME = 34
    MAGIC_ACTIVITY_MUSIC = 35
    SEARCH_MUSIC_RECOMMEND_LIST = 36
    SEARCH_USER_RECOMMEND_LIST = 37
    SEARCH_CHALLENGE = 38
    SEARCH_MUSIC = 39
    SEARCH_USER = 40
    NOTICE_LIST = 41
    COLLECTION = 42
    COMMERCE_FIND_SECOND_FLOOR_AWEME = 43
    COMMERCE_FIND_SECOND_FLOOR_USER = 44
    FOLLOW_FEED_USER = 45
    COMMERCE_CHALLENGE_RELATED_ACCOUNT = 46
    NOTICE_DIGG_LIST = 47
    USER_RECOMMEND = 48
    DISCOVERY_MUSIC_LIST = 49
    HOTSEARCH_VIDEO_LIST = 50
    MISC_AWEME = 51
    MIX_DETAIL = 52
    MIX_COLLECTION = 53
    MIX_AWEME = 54
    MIX_PUBLISH = 55
    MIX_WEB = 56
    NEARBY_FEED = 57
    MIX_AWEME_WEB = 58
    REACT_INFO = 59
    NATIONAL_TASK = 60
    DIGG_LIST = 61
    FORWARD_LIST = 62
    MUSIC_STICKER = 63
    FAVORITE_LIST = 64
    CHANNEL_FEED = 65
    PUBLISH_SEARCH = 66
    FAMILIAR_FEED = 67
    COMMERICAL_MUSIC_LIST = 68
    COMMERICAL_USER_MUSIC_COLLECT = 69
    COMMERICAL_MUSIC_DETAIL = 70
    PRIVATE_PUBLISH = 71
    FAMILIAR_VIDEO_VISITOR_LIST = 72
    CREATOR_CENTER = 73
    OPEN_NEWS = 74
    FRIEND_FEED = 75
    FAMILIAR_UNREAD_USER_LIST = 76
    FAMILIAR_UNREAD_VIDEO_LIST = 77
    HOT_MUSIC_LIST = 78
    WEBCAST_ROOM_PACK = 79
    IM_GROUP_USER_LIST = 80
    USER_PROFILE = 81
    ARK_ANCHOR_DETAIL = 82
    LONG_VIDEO_FEED = 83 // long video recommend feed
    LONG_VIDEO_RELATIVE_FEED = 84 // related long video recommend feed
    UNFOLLOW_CONTACT = 85
    DOU_DISCOUNT_LIST = 86
    USER_SUGGEST = 87 // For You
    TIKTOK_ANALYTICS = 88 // TikTok Analytics
    TIKTOK_DEBUG_LANDING_PAGE = 89 // TikTok debug platform landing page
    TIKTOK_ULIKE = 90 //TIKTOK Cut the same template
    FORWARD_LIST_V2 = 91
    LONG_VIDEO_DETAIL = 92
    KEFU_USER = 93 // Customer service platform user information center
    MUSIC_DISCOVERY = 94 // discover more sounds on music detail page
    MOMENT_AWEME = 95 //moments
    TIKTOK_REFLOW = 96 // TIKTOK reflow page
    MISSION_PLATFORM_AWEME = 97 // Task desk my submission video page
    TIKTOK_DONT_SHARE_LIST = 98 // TikTok publish dont share list
    PROFILE_EFFECT_TAB = 99 // Personal page props list associated video request
    LIVE_TASK = 100 // live task
    COMMON_RELATION_LIST = 101 // Follow / fan list & general relationship (pull black, follow request, etc.) list
    NEW_STORY_20 = 102 // 2020 new fast record video
    TIKTOK_STORY_GUIDE = 103
    BCF = 104 // BCF Video List
    STATUS_FEED = 105 // TIKTOK LITE STATUS FEED
    FAMILIAR_ACTIVITY = 106
    WELFARE = 107 // public welfare
    FPROJECT_B_PROFILE // F project B end personal home page
    SHORT_VIDEO_RELATIVE_FEED = 109 // related short video feed
    HPROJECT_D_QUERY = 110; // H project query
    OPERATION_TASK_AWEME = 111; // TIKTOK Operation task video ranking list
    OPERATION_TASK_DETAIL = 112; // TIKTOK Operation task details page
    CHART_MUSIC_LIST = 114 //TIKTOK Music List
    HORIZONTAL_FEED = 115 // feed Long video inner streaming page
    MUSIC_LONG_VIDEO = 116 // sound page scene for long video
    TIKTOK_TV_FEED = 117 // tiktok tv feed
    TIKTOK_TV_FOLLOW_FEED = 118 // tiktok tv follow feed
    TIKTOK_BA_CREATIVE_HUB = 119 // tiktok ba creative hub page
    TIKTOK_EOY_REVIEW = 120 // tiktok eoy review script
    TIKTOK_SHOUTOUTS = 121 // tiktok shoutouts video
    TIKTOK_OFFLINE_FEED = 122  // TikTok Lite Offline Mode Feed
    TIKTOK_MIX_ITEM_LIST = 123 // ignore packing mix_info when packing items
    TIKTOK_SUGGESTED_ACCOUNT = 124 // TikTok onboarding process, suggested accounts to follow
    TIKTOK_STORY_DETAIL = 125  // tiktok story detail page
    TIKTOK_STORY_ARCHIVE = 126 // tiktok story archive scene
    TIKTOK_FOR_DEVELOPER = 127 // tiktok open platform for developer
    POI_VIDEO = 128 // POI video list
    TIKTOK_FORUM_QUESTION = 129 // tiktok forum q&a
    TIKTOK_BA_BRANDED_CONTENT = 130 // tiktok ba branded content page
    TIKTOK_EOY_MINT = 131 // tiktok EOY campaign for mint platform
    TIKTOK_EOY_DETAIL = 133 // tiktok EOY campaign for client
    TIKTOK_CLA_CROWDSOURCING = 134 // tiktok cla crowdsourcing
    TIKTOK_DUET_DISCOVER = 135 // tiktok duet discover page for top ranking duet videos
    TIKTOK_DSP_COLLECTION = 136 // tiktok dsp clollection
    TIKTOK_DSP_FEED = 137 // tiktok dsp feed
    TIKTOK_FRIEND_FEED = 138 // tiktok friend tab feed
    BRAND_SAFETY = 139 // brand safety video maintainance
    TIKTOK_TRENDS_INFLOW = 140 // tiktok trends inflow page
    TIKTOK_CITY_PAGE = 141 // tiktok city page platform
    TIKTOK_PROFILE_VIEWER = 142 // tiktok profile viewer
    TTEC_FEED = 143 // ttec feed
    TIKTOK_REUSE_ORIGINAL_SOUND = 144 // tiktok reuse original sound
    TIKTOK_CONTENT_SYSTEM = 145 // tiktok content system
    TIKTOK_VIDEO_VIEWER = 146 // tiktok video viewer
    TIKTOK_TTS_VOICE_DETAIL = 147  // tiktok tts voice detail aggregation page, https://bytedance.feishu.cn/docx/doxcn30htafG6BCWpSBDBm7oImb#doxcnowEaakao2GWAkPRXJF5xWb
    GLOBAL_LIVE_ACTIVITY = 148 // global live use video
    VS_ECOSYSTEM = 149 // vertical solutions ecosystem
    POPULAR_FEED = 150 // popular feed
    TT4D_PLAYKIT_FEED = 151 // TikTok Open Platform PlayKit
    TIKTOK_AFFILIATE_VIRAL_VIDEO = 152 // TikTok Affiliate viral video
    PROMOTE_AD_PREVIEW = 153 // tiktok promote ad preview
    TIKTOK_PAID_COLLECTION = 154 // videos belonging to a paid collection
    TIKTOK_BOOKTOK_AWEME = 155 // booktok video list
    STICKER_DISCOVER = 156 // sticker/effect discover
    TIKTOK_CLA = 157 // tiktok cla
    TIKTOK_NOW_FEED = 158 // tiktok now feed
    TIKTOK_NOW_ARCHIVE = 159 // tiktok now archive
    TIKTOK_POI_DETAIL = 160  // poi detail page
    TIKTOK_NOW_EXPLORE_FEED = 161 // tiktok now explore feed
    TIKTOK_ALLIGATOR_EXPLORE_FEED = 162 // tiktok alligator explore feed
    TIKTOK_ALLIGATOR_FRIEND_FEED = 163 // tiktok alligator friend feed
    TIKTOK_ALLIGATOR_PROFILE = 164 // tiktok alligator profile
    TIKTOK_ALLIGATOR_NOTICE = 165 //tiktok alligator normal notice
    TIKTOK_CASCADED_ITEM_LIST = 166 // duet/stitch item list
    TIKTOK_LIVE_SUBSCRIPTION = 167
    OTHERS_COLLECTION = 168 // sharing favorites
    TIKTOK_ALLIGATOR_AWEME_DETAIL = 169 // tiktok alligator get aweme detail
    TIKTOK_BA_ANALYTICS = 170 // tiktok BA analytcs
    TIKTOK_ALLIGATOR_WIDGET = 171 // tiktok alligator widget
    TIKTOK_UPVOTE = 172 // tiktok upvote items
    TIKTOK_ACTIVITY_CENTER = 173 // tiktok activity center
    TIKTOK_UG_MATERIAL = 174 // tiktok ug material
    TIKTOK_AUDIO_FEED = 175 // tiktok audio feed
    TIKTOK_NOW_USER_POST = 176 // tiktok now user post
    CC_TEMPLATE_MUSIC_DETAIL = 177 // add new pack source for cc template in music detail page
    TIKTOK_MOVIETOK_AWEME = 178 // movietok video list
    TIKTOK_TOPIC_FEED = 179 // tiktok topic feed
    TIKTOK_VIDEO_STICKER_CREATE_FEED = 180 // pack source for creating video sticker from sticker store
    SEARCH_APP_ENGINE = 181 // tiktok search app engine
    TIKTOK_EXPLORE_FEED = 182 // tiktok explore feed
    SNIPPETS_VIDEO_INFO = 183 // snippets_video_info
    TIKTOK_INBOX_SKYLIGHT = 184 // tiktok inbox skylight
    TIKTOK_CREATOR_CENTER_VIDEO_LIST = 185 // tiktok creator center creator video list
    TIKTOK_AD_AWEME_DETAIL = 186 // for topview ads 、tip ads or others ads video
    TIKTOK_IM_CHAT_BOT = 187 // for im tikbot scene; pack answer's video info
    TIKTOK_SUB_ONLY_VIDEOS_LIST = 188 // for sub only videos list info
    TIKTOK_IM_VIDEO = 189 // for IM share video
    TIKTOK_VC_FILTER_DETAIL = 190 // for SAMI voice change filter detail page; Tech design: https://bytedance.us.feishu.cn/docx/F25ddGslWobgF6xizhuuJHB2spe
    TIKTOK_SUB_ONLY_VIDEOS_LIST_BRIEF = 191 // for sub only videos list brief info, without isSubscriber checker
    TIKTOK_DANMAKU_LIST = 192 // for video danmaku list user relation and status check
    TIKTOK_TRAFFIC_INCENTIVE = 193,          // For traffic incentive video list
    TIKTOK_ADDYOURS_TOPIC_DETAIL = 194 // for add yours topic detail
    MIX_COLLECT_LIST = 195 // for mix/collect_list interface; pack video cover info
    TIKTOK_PODCAST_PUBLISH_PAGE = 196 // for the user's published videos in the podcast publish/edit page
    TIKTOK_LERT_TSET = 197 // for PnS LERT/TSET DSA requirements; more info: https://bytedance.feishu.cn/wiki/wikcnteX9diSE5n2w6ELMSSpKFf
    TIKTOK_CREATOR_CENTER_NOTIFICATION = 198 // For Creator center notification platform: https://bytedance.feishu.cn/wiki/wikcnhLzl55BKga4ps4avftDUVc
    TIKTOK_BATCH_MODIFY_VISIBILITY_ITEM_LIST = 199 // For Activity Center batch modify visibility item list: https://bytedance.feishu.cn/docx/OeuVdQG4SoNR3RxDE7BcRSubnnd
    TIKTOK_GAME_ANCHOR_FEED = 200 // For same game anchor videos list brief info
    TTM_SEARCH = 201 // For tiktok music pack tt video in search scenario.
    TTM_CRUSH_MODE = 202 // For tiktok music pack tt video in Crush mode.
    TIKTOK_TV_CASTING = 203 // For tiktok tv casting mode.
    TIKTOK_TRASH_BIN = 204 // For tiktok trash bin. More info: https://bytedance.larkoffice.com/wiki/C91qwSjnli9slHkqz1mcjUUYnHd
    TIKTOK_EVERGREEN_PROFILE_BANNER = 205 // For tiktok evergreen profile banner. See https://bytedance.us.larkoffice.com/docx/M9kud1hH5o1sJnxMI9murQB0szg
    TIKTOK_SHARE_FEED = 206 // tiktok share interaction feed, related doc: https://bytedance.larkoffice.com/docx/L4IUdXHjsoh6Waxeyz3cKphDnag
    TIKTOK_SPARK_APP = 207 // tiktok spark app
    TIKTOK_SPARK_APP_FEED = 208 // tiktok spark app feed
    TIKTOK_SPARK_APP_FOLLOWING = 209 // tiktok spark app following page
    TIKTOK_SPARK_APP_PROFILE = 210 // tiktok spark app profile page
    TIKTOK_SPARK_APP_SEARCH = 211 // tiktok spark app search page
    TIKTOK_DM_CARD = 222 // tiktok shared message card in DM. More info: https://bytedance.larkoffice.com/wiki/N2lOwY0uQiLAUIkakW8cWrJ4n2b}
    TIKTOK_FOLLOW_SKYLIGHT = 223 // tiktok follow skylight
    TIKTOK_AFFILIATE_MERCHANT_CRM = 224 // TikTok Affiliate Merchant CRM
    TIKTOK_LITE_MARKETPLACE = 225 // For tiktok lite marketplace
    TIKTOK_MY_MUSIC = 226 // tiktok my music page. More info: https://bytedance.larkoffice.com/wiki/Dn8MwrlFHiHoIUkRjnPcJ13ankg
    TIKTOK_HORIZONTAL_FEED = 227 // tiktok horizontal feed. More info: https://bytedance.sg.larkoffice.com/docx/PbFFd6HPUoNETexXo0xcPerTnph
    TIKTOK_SNAIL_IM_VIDEO = 228 // for snail share video
    TIKTOK_SNAIL_SEARCH = 229 // For snail search
    TIKTOK_SNAIL_SEARCH_TIKTOK = 230 // for search tiktok content in snail
    TIKTOK_SNAIL_INBOX_TIKTOK_DETAIL = 231 // for search tiktok content in snail
    TTEC_FEED_WITH_LIVE = 232 // ttec feed with live status.More info: https://bytedance.sg.larkoffice.com/docx/BBwodbH5Xo7XK2xVwNKly4C1gOh
    MUSICVERSE_ONE_ID = 233 // musicverse one id
    TIKTOK_SEARCH_INSPIRATION = 234 // for tiktok search inspiration pack specific videos
    CREATION_INSPIRATION = 235 // for creation inspiration. More info: https://bytedance.sg.larkoffice.com/docx/HMdwd8CZcoWlF1xSGtacrD0onNg
    CREATION_INSPIRATION_NON_LOGGEDIN = 236 // for creation inspiration's non logged-in users. More info: https://bytedance.sg.larkoffice.com/docx/UsVbdOqREo7qp2xmnI7li7LAgvd
    TIKTOK_SIMILAR_FEED = 237 // tiktok similar feed. More info: https://bytedance.sg.larkoffice.com/docx/Lk1pdmBrnouShmxhaDqlSQoAgC4
    TIKTOK_BULK_MANAGE_COMMENT_PERMISSION = 238 // For tiktok bulk modify posts' comment permission More info:https://bytedance.larkoffice.com/wiki/JxiYwOnIpiJbt8k6qygcLYtsnfh
    TIKTOK_ECOMMERCE_FEED = 239 // tiktok related ecommerce video feed. More Info: https://bytedance.larkoffice.com/docx/PaicdJmZboo0d4xjLkMcuXHznQb
    TIKTOK_PAID_ITEM_UNLOCK_REFETCH = 240 // tiktok paid item refetch after unlock
    TIKTOK_PLAY_MODE_FEED = 241 // tiktok playMode feed. More info: https://bytedance.sg.larkoffice.com/docx/ETt2d2cj4owAj0xeagwlF1LzgVf
}

enum CommerceActivityTypeEnum {
    GESTURE_RED_PACKET = 1
    PENDANT = 2
}

enum ConstellationTypeEnum {
    UNKNOWN = 0
    Deprecated1 = 1
    Deprecated2 = 2
    Deprecated3 = 3
    Deprecated4 = 4
    Deprecated5 = 5
    Deprecated6 = 6
    Deprecated7 = 7
    Deprecated8 = 8
    Deprecated9 = 9
    Deprecated10 = 10
    Deprecated11 = 11
    Deprecated12 = 12
}

enum FavoriteListPermissionEnum {
    ALL_SEE  = 0
    SELF_SEE = 1
}

enum FollowStatusEnum {
    UNKNOWN = -1
    NONE = 0
    FOLLOW = 1
    MUTUAL = 2
    FOLLOWING_REQUEST = 4
}

enum GameTypeEnum {
    CATCH_FRUIT = 1; # catch fruit game
}

enum Deprecated1Enum {
    UNKNOWN = 0
    Deprecated1 = 1
    Deprecated2 = 2
}

enum InteractionTypeEnum {
    NONE = 0
    Deprecated1 = 1
    VOTE = 3
    COMMENT = 4
    WIKI = 5
    DONATION = 6
    Deprecated7 = 7
    MENTION = 8
    HASH_TAG = 9
    LIVE_COUNTDOWN = 10
    AUTO_CAPTION = 11
    //12->15 are reserved - keep consistent with IDL file
    DUET_WITH_ME = 16
    FORUM = 17
    TEXT_STICKER = 18
    SHARE_TO_STORY = 19
    BURN_IN_CAPTION = 20
    POI = 22
    NATURE_CLASSIFICATION = 23
    MUSIC_STICKER = 42
    AI_SONG = 50
    ADD_YOURS = 88
}

enum LyricTypeEnum {
    NONE = 0
    TRC = 1
    LRC =2
    KRC =3
    TXT = 4
    JSON = 10
    PREVIEW_TXT = 11
}

enum MediaTypeEnum {
    NONE = 0
    TEXT = 1
    PIC = 2
    GIF = 3
    VIDEO = 4
    PIC_LIST = 5
    STORY = 11
    VR = 12
    FORWARD = 21
    STORY_LIVE = 22
    STORY_PIC = 23
    MUSIC_DSP = 41
}

enum MixTypeEnum {
    NORMAL = 0
}

enum MaskTypeEnum {
    REPORT = 1
    DISLIKE = 2
    GENERAL = 3   // general mask
    PHOTOSENSITIVE = 4  // photosensitie epilepsy
    CONTENT_CLASSIFICATION = 5 // content classification mask
}

enum RelationLabelTypeEnum {
    FOLLOWING = 0;
    FROWARD = 1;
    DIGG_LIST = 2;
    FORWARD_WITH_AVATAR = 3;
    COMMENT_WITH_AVATAR = 4;
    RECOMMEND_FOLLOW = 5;
    SUGGESTED_ACCOUNT = 6;    // MT label: Suggested account
    FOLLOWED_BY_SINGLE = 7;   // MT label: Followed by %username
    FOLLOWED_BY_MULTI = 8;    // MT label: Followed by %username + n others
    YOUR_CONTACT = 9;         // MT label: Your contact
    FROM_YOUR_CONTACTS = 10;  // MT label: From your contacts
    PEOPLE_MAY_KNOW = 11;     // MT label: People you may know
    YOUR_CONNECTIONS = 12;    // MT label: Your connections
    CONNECTED_TO_YOU = 13;    // MT label: Connected to you
    SUGGESTED_FOU_YOU = 14;   // MT label: Suggested for you
    FOLLOWED_BY_WITH_AVATAR = 15;
    TT_FOLLOWING = 18;         // MT label: Following
    TT_RECENTLY_FOLLOWED = 19; // MT label: Recently followed
    TT_YOUR_FRIEND = 20;       // MT label: Your friend
    DIGG_LIST_WITH_AVATAR = 22; // digg list
    TT_FOLLOWS_YOU = 23; // MT label: Follows you
    TT_TWITTER_CONNECTION = 24; // MT Label: Connected on Twitter
}

enum ForumItemTypeEnum {
    FORUM_UNKNOWN = 0;
    FORUM_QUESTION = 1;
    FORUM_ANSWER = 2;
}

enum TextTypeEnum {
    USER_TEXT = 0
    CHALLENGE_TEXT = 1
    COMMENT_TEXT = 2
}

enum VerficationTypeEnum {
    NONE = 0
    DEFAULT = 1
    ORIGINAL_MUSICIAN = 2
}

enum VideoDistributeTypeEnum {
    SHORT_VIDEO = 1              // short video
    LONG_VIDEO_DIRECT_PLAY = 2   // long video
    LONG_VIDEO = 3               // long video with short video
}

enum MusicUnusableReasonTypeEnum {
    USABLE = 0
    COPYRIGHT = 1
    OTHER = 2
}

enum UserStoryStatus {
    No_Visible_Story = 0;
    Has_Visible_Story = 1;
    All_Viewed_Visible_Story = 2;
}

enum MusicVideoStatusEnum {
    VIDEO_LOOSE_NOT_RECOMMEDN = 150
    VIDEO_STRICT_NOT_RECOMMEDN = 170
    VIDEO_NOT_RECOMMEND = 200
    VIDEO_MUTE = 250
    VIDEO_MUTE_AND_NOT_RECOMMEND = 255
    VIDEO_MUTE_AND_PROFILE_SEE  = 258
    VIDEO_SELF_SEE_EXPT_OWNER = 260
    VIDEO_SELF_SEE = 300
    VIDEO_MUTE_AND_SELF_SEE = 350
    VIDEO_DEL_EXPT_OWNER = 400
    VIDEO_DELETE = 1000
}

enum MusicArtistTypeEnum {
    ORIGINAL = 1 // ORIGINAL musician
    CELEBRITY = 2 // KOL CELEBRITY
}

enum MultiBitRateMusicEnum {
    MusicBitRateMedium64Enum = 3
    MusicBitRateHigher128Enum = 6
    MusicBitRateHighest256Enum = 7
}

enum MTCertTypeEnum {
    EMPTY        = -1,  // unverified
    PERSONAL     = 0,   // Tiktok: Personal
    ORGANIZATION = 1,   // TikTok: Non-Profit Organization
    BUSINESS     = 2,   // TikTok: Business/Brand
}

enum ImprTagEnum {
    None = 0,
    ChangQing = 1,
    HorizonVideo =2,
}

//  0.personal account 1.pro account  2.creator account 3.business account
enum ProAccountStatus {
    PROACCOUNT_CLOSE = 0,
    PROACCOUNT_OPEN = 1,
    CREATORACCOUNT_OPEN = 2,
    BUSINESSACCOUNT_OPEN = 3,
}

enum Deprecated1TypeEnum {
    NO_NEED = 0,
    GENERAL = 1,
    FORCE_OP = 2,
}

enum MusicUnshelveReasonEnum {
    UNKNOWN = 0,
    COPYRIGHT = 1,
    COMMUNITY_SAFETY  = 2,
    TRUST_AND_PRIVACY = 3
}

enum TCMReviewStatusEnum {
    InReviewing = 1;
    ReviewReject = 2;
    ReviewPassed = 3;
}

enum BottomBarTypeEnum {
    LEARN_FEED = 0
}

enum BoostTypeEnum {
    BOOST_UNKNOWN = 0
    BOOST_MUSIC = 1
}


enum AutoCaptionLocationType {
    LEFT_TOP = 0
    MIDDLE_TOP = 1
    RIGHT_TOP = 2
    LEFT_MIDDLE = 3
    MIDDLE_MIDDLE = 4
    RIGHT_MIDDLE = 5
    LEFT_BOTTOM = 6
    MIDDLE_BOTTOM = 7
    RIGHT_BOTTOM = 8
}

enum BlockStatusEnum {
    UNKNOWN = -1
    NONE = 0
    FROMME = 1
    TOME = 2
    MUTUAL = 4
}

enum SoundExemptionStatus {
    NO_EXEMPTION = 0
    SELF_PROMISE_EXEMPTION = 1
}

enum GreenScreenType {
    UNKNOWN_GREEN_SCREEN = 0
    PHOTO_GREEN_SCREEN = 1
    VideoGreenScreen   = 2
    GIFGreenScreen     = 3
}

enum InteractPermissionEnum {
    ABLE = 0
    DISABLE = 1
    HIDE = 2
    DISABLE_FOR_ALL = 3
    HIDE_FOR_ALL = 4
}

// for notice UI template
enum TextLinkTypeEnum { // TextLink UI type
    UNDEFINED = 0; // undefined
    PLAIN_TEXT = 1; // common text
    BOLD_LINK = 2;  // bold + link
}

// for notice UI template
enum TextLinkSchemaTypeEnum {
    UNDEFINED = 0; // undefined
    PROFILE = 1; // for profile page
}

enum MutualType {
    MUTUAL_TYPE_FRIEND = 1;
    MUTUAL_TYPE_FOLLOWED = 2;
    MUTUAL_TYPE_FOLLOW = 3;
}

enum InteractionTagInterestLevel {
    VIDEO_TAG_INTEREST_LEVEL_UNDEFINED = 0;
    VIDEO_TAG_INTEREST_LEVEL_LOW = 1;  // FIND NOT INTEREST IN THIS TAG
    VIDEO_TAG_INTEREST_LEVEL_HIGH = 2;  // FIND PEOPLE IN IT ARE RELATED TO ONESELF
}

enum NoticeButtonType {
    NoticeButtonType_Normal = 0; // white background
    NoticeButtonType_Follow = 1;
    NoticeButtonType_Red = 2; // red background
    NoticeButtonType_QuickCommentLikeReply = 3; // 2 buttons
    NoticeButtonType_CommentAddAsPost = 4;
}

enum NoticeButtonActionType {
    NoticeButtonActiontype_Unknown = 0
    NoticeButtonActionType_Delete = 1; // delete notice
    NoticeButtonActionType_CommentAddAsPost = 2;
}

enum MusicCommercialRightType {
    MusicCommercialRightType_Unknown = 0;
    MusicCommercialRightType_Noncommercial = 1;
    MusicCommercialRightType_General = 2;
    MusicCommercialRightType_Private = 3;
}

enum SpecialAccountType {
    AdFake = 1
    Staff = 2
    TTMusic = 3
    TTSeller = 4
    TTLiveOperator = 5
    TestAccount = 6
    TTNow = 7
    TTAlligator = 8
    CapCutTemplateVirtual = 9
    TTShopOpenTest = 10
    TTMint = 11
    TtcmShadow = 12
}

enum AccountLogStatusEnum {
    Default = 0
    NotLoggedIn = 1
    HasLoggedIn = 2
}

enum PoiContentExpType {
    DirectExpand   = 0
    TailExpand     = 1
    HeadExpand     = 2
}

enum PoiAnchorContentType {
    FriendPostCNT  = 0
    PeopleFavCNT   = 1
    PeoplePostCNT  = 2
    VideoCNT       = 3
    Category       = 4
    Guidance       = 5
}

enum NowPostSource {
    NowPostSourceUnknown = 0;
    NowPostSourceFriends = 1;
    NowPostSourceFollowing = 2;
    NowPostSourcePopular = 3;
}

enum PoiReTagType {
    NoNeed  = 0
    General = 1
}

enum NowViewState {
    NowNotViewed = 0;
    NowViewedFuzzy = 1;
    NowViewedVisible = 2;
}

enum PodcastFollowDisplay {
    FollowNone = 0; // not show follow button on audio feed
    Follow = 1; // show follow button on audio feed
}

enum NowCollectionType {
  NowCollectionTypeFlatted = 1;
  NowCollectionTypeAggregated = 2;
}

enum StoryType {
    NotUsed = 0;
    Default = 1;
    TextPost = 2;
    StoryTypeNowStyle = 3;
    StoryTypeAIGCPost = 4;
}

enum UserNowStatus {
  NoVisibleNow = 0;
  HasVisibleNow = 1;
  ViewedVisibleNow = 2;
}

enum DiggShowScene {
  DiggShowNormal = 0;
  HideDiggCount = 1;
}

enum TTMProductType {
    Resso = 1;
    TTMusic = 2;
}

enum EditPostBiz {
    TEXT         = 0  // Description
    COVER        = 1  // Cover
    POI          = 2  // POI
    LEMON8_MUSIC = 3  // Lemon8 music
    // CML music
	CML_MUSIC = 4;
	// Ad Setting - Disclose Branded Content
	DISCLOSE_BRANDED_CONTENT = 5;
	// Ad Setting - Disclose Brand Organic (aka. "Your Brand")
	DISCLOSE_BRAND_ORGANIC = 6;
	// allow comment
	ALLOW_COMMENT = 7;
	// allow duet
	ALLOW_DUET = 8;
	// allow stitch
	ALLOW_STITCH = 9;
	// add post to playlist
	PLAYLIST = 10;
	// visibility
	VISIBILITY = 11;
	// the overall edit-post entry
	OVERALL = 12;
	// delete
	DELETE = 13;
}
enum EditPostControlStatus {
    NO_SHOW                         = 0  // show no entrance for edit_post
    SHOW                            = 1  // show entrance for edit_post
    GRAYED_OUT_FOR_MODERATION       = 2  // show grey entrance and toast "in moderation"
    GRAYED_OUT_FOR_FREQ_CONTROL     = 3  // show grey entrance and toast "cause frequncy control", such as there is only one chance each day to edit post
    GRAYED_OUT_FOR_TIME_EXPIRATION  = 4  // show grey entrance and toast "cause time limit", such as only video posted within 7 days could be edited
    Grayed_Out_For_In_Process       = 5
    // show grey entrance
    GRAYED_OUT = 6;
}

enum EditPostControlReason {
	// default
	DEFAULT = 0;
	// the original post is under initial review
	UNDER_INITIAL_REVIEW = 1;
	// ad post
	AD_ITEM = 2;
	// branded-content post
	BRANDED_CONTENT_ITEM = 3;
	// mission post
	BRANDED_MISSION_ITEM = 4;
	// promoting post
	IS_PROMOTING_ITEM = 5;
	// allow-comment shown with description for ad item
	ALLOW_COMMENT_HAS_AD_DESCRIPTION = 6;
    // schedule post that hasn't been published yet
	SCHEDULE_ITEM = 7;
	// photomode item
	PHOTOMODE_ITEM = 8;
	// commercial item
    COMMERCIAL_ITEM = 9;
    // cover being synchronized
    COVER_BEING_SYNCHRONIZED = 10;
    // post under content check
    UNDER_CONTENT_CHECK = 11;
}

enum NowAvatarBadgeStatus {
  NoNowAvatarBadge = 0; // no now avatar badge
  ChristmasNewYearAvatarBadge = 1; // christmas new year avatar badge
  NonChristmasNewYeaAvatarBadge = 2; // now christmas new year avatar badge
}

enum ForcedVisibleState {
   Ignore = 0;
   Visible = 1;
 }

enum InteractionAction {
    Ignore = 0; //走线上逻辑，兼容原来逻辑，客户端判断应该触发什么
    DiggAction = 1; //端上走点赞动作流程
    EnterCommentPannel = 2; //打开评论面板
    EnterShoot = 3; //跳转拍摄页
}

enum NowBlurType {
    Ignore = 0;
    FrontVisible = 1; //前置清晰
    Gauss = 2; //高斯模糊，图片投稿为前置高斯，视频投稿为合成图高斯
}

enum NowPostCameraType {
    DefaultDual = 0;    // 默认双摄
    Rear = 1;           // 后置摄像头拍摄
    Front = 2;          // 前置摄像头拍摄
}

enum InteractiveMusicStreamingDetailStatusEnum {
    ALL_USEABLE = 100 // available
    UNFEEDALBE = 1000
    UNPLAYABLE = 2000
    INVISIBLE = 3000
    UNUSEABLE = 10000 //Unavailable
}

enum ClaCaptionsType {
   ClaCaptionsTypeUnspecified = 0; // means no captions are provided (caption_infos is empty)
   ClaCaptionsTypeOriginal = 1; // only have one caption which is original language
   ClaCaptionsTypeOriginalWithTranslation = 2; // original and translated captions are provided
 }

enum ClaOriginalCaptionType {
  ClaOriginalCaptionTypeUnspecified = 0;
  ClaOriginalCaptionTypeAsr = 1;
  ClaOriginalCaptionTypeCec = 2;
  ClaOriginalCaptionTypeStickerCreator = 3;
  ClaOriginalCaptionTypeCapcutCreator = 4;
  ClaOriginalCaptionTypeThirdPartyCreator = 5;
  ClaOriginalCaptionTypeClosedCreator = 6;
}

enum ClaNoCaptionReason {
    ClaNoCaptionReasonNoOp = 0;
    ClaNoCaptionReasonOther = 1;
    ClaNoCaptionReasonNotAuthorized = 2; // Disabled by creator
    ClaNoCaptionReasonSpeechUnrecognized = 3; // SubtitleStatus_UNQUALIFIED_CAPTION
    ClaNoCaptionReasonLangNotSupported = 4; // SubtitleStatus_UNSUPPORTED_LANGUAGE
}

enum SerializableItemType {
      Video = 0;      // 默认类型：短视频
      Ad = 1;           // 广告类型
      Live = 2;          // 直播类型
 }

 enum DspPlatform {
   DspPlatform_Unknown = 0, // should not appear, for the zero value issue
   DspPlatform_AppleMusic = 1,
   DspPlatform_AmazonMusic = 2,
   DspPlatform_Spotify = 3,
   DspPlatform_TTMusic = 4,
   DspPlatform_Melon = 7,
   DspPlatform_Deezer = 8,
 }

enum TTMUaSwitchStatus {
  TTMBetaSwitchOff = 1,
  TTMBetaSwitchOn = 2,
}

enum RedirectPage{
  GeneralSearchPage = 0;
  EcomSearchPage = 1;
}

 enum TaggingBaInvitationStatus {
    Unknown = 0;
    Pending = 1;
    Accept = 2;
 }

enum AudioChangeStatusEnum {
    AudioUnchangeable = 0;
    AudioChangeable = 1;
    AudioChanged = 2;
}

enum DetailStatusReasonTypeEnum {
    UNKNOWN = 0,
    COPYRIGHT = 1,
    COMMUNITY_SAFETY = 2,
    TRUST_AND_PRIVACY = 3
}


enum ShortVideoDetailStatusEnum {
    ALL_USEABLE = 100 // Available and discoverable
    USEABLE_BUT_UNDISCOVERABLE = 200 // Available but not discoverable
    OWNER_USEABLE = 1000 //Used by yourself
    UNUSEABLE = 10000 //Unavailable
}

enum PoiHideType {
    Suffix  =   1000
}

enum NoticeActionEnum {
    NOTICE_ACTION_UNKNOWN = 0;
    NOTICE_ACTION_CLICK = 1;
    NOTICE_ACTION_READ = 2;
}

enum NoticeBehaviorActionEnum {
    NOTICE_BEHAVIOR_UNKNOWN = 0;
    NOTICE_BEHAVIOR_REPORT = 1;
}

enum NoticeOperationTypeEnum{
    NOTICE_OPERATION_TYPE_UNKNOWN = 0;
    NOTICE_OPERATION_TYPE_DELETE = 1;
}

enum InsightStatusEnum {
  InsightStatus_NA = 0; // feature not available, should show old version button
  InsightStatus_Success = 1; // user opted in, data available
  InsightStatus_NotOptedIn = 2; // data not available due to user not opted in
  InsightStatus_Inactive = 3; // data not available due to inactivity
}

enum PodcastTnsStatus {
    TNS_STATUS_UNKNOWN = 0
    TNS_STATUS_APPROVED = 1
    TNS_STATUS_REJECTED = 2
    TNS_STATUS_NOT_RECOMMENDED = 3
}

enum PodcastSharkStatus {
    SHARK_STATUS_UNKNOWN         = 0
	SHARK_STATUS_PASS            = 1
	SHARK_STATUS_BLOCK_INVISIBLE = 2
	SHARK_STATUS_BLOCK_REJECT    = 3
}

enum TrendingProductRecallType {
    SAME             = 1
	SAME_AND_SIMILAR = 2
	SIMILAR          = 3
	CATEGORY         = 4
	HaveEcomIntent   = -1
    NoEcomIntent     = -2
}

enum MentionStickerScenario {
    MENTION_STICKER = 0
    TEXT_STICKER    = 1
    STORY_TO_STORY  = 2
}

enum AvatarDefaultShowEnum {
    Default = 0,
    URAvatar = 1,
}

enum AvatarSourceEnum {
    Default = 0,
    SelfUploading = 1,
    AIGC = 2
}

enum AvatarCategory {
    Default = 0,
    Static = 1,
    Dynamic = 2,
    Local_Static = 3,
    Local_Dynamic = 4,
    Navi = 5,
    Create_Initial = 6,
    Static_UR = 7,
    Dynamic_UR = 8,
    UR_Without_BG = 9,
    Local_Static_UR = 10,
    Local_Dynamic_UR = 11,
    Local_UR_Without_BG = 12,
}
