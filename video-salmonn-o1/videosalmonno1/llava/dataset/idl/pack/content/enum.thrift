namespace go tiktok.content.enum


// 广告授权状态
enum AdAuthStatus {
	// 不可授权/授权已过期
	AD_AUTH_DISABLED = 0;
	// 未授权
	AD_AUTH_UNAUTHORIZED = 1;
	// 已授权
	AD_AUTH_AUTHORIZED = 2;
	// 已拒绝
	AD_AUTH_DENIED = 3;
}

// 广告来源 (授权类型)
enum AdSource {
	// 用户发布/取物料
	USER_PUBLISH = 1;
	// 广告主发布/塞物料
	ADVERTISER_PUBLISH = 2;
}

enum AddYoursTopicTypeEnum {
	Unknown = 0;
	OperationalRecommendation = 1;
	AIImage = 2;
	PhotoText = 3;
	Hashtag = 4;
}

// addyours trend video source
enum AddYoursTrendVideoSourceEnum {
	UnknownSource = 0;
	Bind = 1;
	AyTrendEntry = 2;
	AySticker = 3;
}

enum AigcLabelType {
	AigcLabelTypeUnknown = 0;
	AigcLabelTypeManual = 1;
	AigcLabelTypeAuto = 2;
}

// item or creator analytics data status used in tiktok sutido; https://bytedance.sg.larkoffice.com/docx/Uuzydj2yeo3jEVxDJDylSWI2gKe
enum AnalyticsDataStatus {
	// success
	SUCCESS = 1;
	// bot pubic
	NOT_PUBLIC = 2;
}

enum AnchorActionType {
	SHOOT = 1;
}

enum AnchorGeneralType {
	GENERAL_ANCHOR = 1;
}

enum AnchorState {
	VISIBLE = 0;
	INVISIBLE = 1;
	SELF_VISIBLE = 2;
}

enum AnchorType {
	// wikipeida
	WIKIPEDIA = 0;
	// MT shop link
	SHOP_LINK = 6;
	// MT yelp anchor
	YELP = 8;
	// MT TripAdvisor anchor
	TRIPADVISOR = 9;
	GAME = 10;
	// Film and television variety show
	MEDIUM = 11;
	MT_MOVIE = 18;
	DONATION = 19;
	RESSO = 23;
	TT_EFFECT = 28;
	TT_TEMPLATE = 29;
	THIRD_PARTY_GENERAL = 44;
	TT_EDIT_EFFECT = 64;
	TICKETMASTER = 69;
}

// The Scenario of Attribution Link Sticker
enum AttributionLinkScenario {
	// unknown
	AttributionLinkScenario_Unknown = 0;
	// music
	AttributionLinkScenario_Music = 1;
	// video
	AttributionLinkScenario_Video = 2;
	// image
	AttributionLinkScenario_Image = 3;
	// link
	AttributionLinkScenario_Link = 4;
}

// The share format of Attribution Link Sticker
enum AttributionLinkShareFormat {
	// unknown
	AttributionLinkShareFormat_Unknown = 0;
	// green screen
	AttributionLinkShareFormat_GreenScreen = 1;
	// photo mode
	AttributionLinkShareFormat_PhotoMode = 2;
}

enum AudioChangeStatusEnum {
	AudioUnchangeable = 0;
	AudioChangeable = 1;
	AudioChanged = 2;
}

enum AutoCaptionLocationType {
	LEFT_TOP = 0;
	MIDDLE_TOP = 1;
	RIGHT_TOP = 2;
	LEFT_MIDDLE = 3;
	MIDDLE_MIDDLE = 4;
	RIGHT_MIDDLE = 5;
	LEFT_BOTTOM = 6;
	MIDDLE_BOTTOM = 7;
	RIGHT_BOTTOM = 8;
}

enum BannerActionType {
	Unknown = 0;
	// 箭头跳转
	ArrowJump = 1;
	// 按钮类选择反馈
	ButtonSelection = 2;
	// 文本类选择反馈
	TextSelection = 3;
	// 关闭按钮类
	Close = 4;
	// Custom Banner Solutions
	Custom = 5;
}

enum BannerTextActionType {
	None = 0;
	SchemaJump = 1;
}

enum BlockStatusEnum {
	NONE = 0;
	UNKNOWN = 1;
	FROMME = 2;
	TOME = 3;
	MUTUAL = 4;
}

enum BottomBarTypeEnum {
	LEARN_FEED = 0;
}

enum CLACaptionsType {
	// means no captions are provided (caption_infos is empty)
	ClaCaptionsTypeUnspecified = 0;
	// only have one caption which is original language
	ClaCaptionsTypeOriginal = 1;
	// original and translated captions are provided
	ClaCaptionsTypeOriginalWithTranslation = 2;
}

enum CLANoCaptionReason {
	ClaNoCaptionReasonNoOp = 0;
	ClaNoCaptionReasonOther = 1;
	// Disabled by creator
	ClaNoCaptionReasonNotAuthorized = 2;
	// SubtitleStatus_UNQUALIFIED_CAPTION
	ClaNoCaptionReasonSpeechUnrecognized = 3;
	// SubtitleStatus_UNSUPPORTED_LANGUAGE
	ClaNoCaptionReasonLangNotSupported = 4;
}

enum CLAOriginalCaptionType {
	ClaOriginalCaptionTypeUnspecified = 0;
	ClaOriginalCaptionTypeAsr = 1;
	ClaOriginalCaptionTypeCec = 2;
	ClaOriginalCaptionTypeStickerCreator = 3;
	ClaOriginalCaptionTypeCapcutCreator = 4;
	ClaOriginalCaptionTypeThirdPartyCreator = 5;
	ClaOriginalCaptionTypeClosedCreator = 6;
}

// celebration type
enum CelebrationType {
	// default value
	CELEBRATION_DEFAULT = 0;
	// birthday celebration
	CELEBRATION_BIRTHDAY = 1;
}

enum CollabStatusEnum {
	NONE = 0;
	PENDING = 1;
	ACCEPT = 2;
	DECLINE = 3;
	EXIT = 4;
	DELETED = 5;
}

enum CommerceActivityTypeEnum {
	GESTURE_RED_PACKET = 1;
	PENDANT = 2;
}

enum DiggShowScene {
	DiggShowNormal = 0;
	HideDiggCount = 1;
}

enum DisplayLayoutEnum {
	VideoSingle = 0;
	ImageSlide = 1;
	// images picture-in-picture, must with 2 images
	ImagePIP = 2;
}

enum DspPlatform {
	AppleMusic = 1;
	AmazonMusic = 2;
	Spotify = 3;
	TTMusic = 4;
}

enum EditPostBiz {
	// Description
	TEXT = 0;
	// Cover
	COVER = 1;
	// POI
	POI = 2;
	// Lemon8 music
	LEMON8_MUSIC = 3;
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

enum EditPostControlStatus {
	// show no entrance for edit_post
	NO_SHOW = 0;
	// show entrance for edit_post
	SHOW = 1;
	// show grey entrance and toast "in moderation"
	GRAYED_OUT_FOR_MODERATION = 2;
	// show grey entrance and toast "cause frequncy control", such as there is only one chance each day to edit post
	GRAYED_OUT_FOR_FREQ_CONTROL = 3;
	// show grey entrance and toast "cause time limit", such as only video posted within 7 days could be edited
	GRAYED_OUT_FOR_TIME_EXPIRATION = 4;
	GRAYED_OUT_FOR_IN_PROCESS = 5;
	// show grey entrance
	GRAYED_OUT = 6;
}

enum FollowStatus {
	// unknown status, as 0 is used for other info
	FollowRelationUnknown = -1;
	// 没有关注对方
	NoRelationStatus = 0;
	// 关注对方
	FollowingStatus = 1;
	// 互关
	FollowEachOtherStatus = 2;
	// 私密账号关注请求
	FollowRequestStatus = 4;
}

enum ForcedVisibleState {
	Ignore = 0;
	Visible = 1;
}

enum GameTypeEnum {
	// catch fruit game
	CATCH_FRUIT = 1;
}

enum GreenScreenType {
	UnknownGreenScreen = 0;
	PhotoGreenScreen = 1;
	VideoGreenScreen = 2;
	GIFGreenScreen = 3;
}

enum HybridLabelTypeEnum {
	Unknown = 0;
	Privacy = 1;
	Friends = 2;
	Following = 3;
	Maf = 4;
}

// Type of conversation/chat in IM domain
enum IMConversationType {
	// invalid default value
	IM_CONVERSATION_TYPE_UNKNOWN = 0;
	// private (2 users directly) chat
	IM_CONVERSATION_TYPE_PRIVATE = 1;
	// group chat
	IM_CONVERSATION_TYPE_GROUP = 2;
}

// Type of video recommended to the feed stream in IM domain
enum IMRecommendType {
	// invalid default value
	IM_RECOMMEND_TYPE_UNKNOWN = 0;
	// video is recommended as it's the originally shared and unread one
	IM_RECOMMEND_TYPE_UNREAD = 1;
	// recommended video similar to the one originally shared
	IM_RECOMMEND_TYPE_SIMILAR = 2;
}

enum InsightStatusEnum {
	// feature not available, should show old version button
	InsightStatus_NA = 0;
	// user opted in, data available
	InsightStatus_Success = 1;
	// data not available due to user not opted in
	InsightStatus_NotOptedIn = 2;
	// data not available due to inactivity
	InsightStatus_Inactive = 3;
}

enum InteractPermissionEnum {
	ABLE = 0;
	DISABLE = 1;
	HIDE = 2;
	DISABLE_FOR_ALL = 3;
	HIDE_FOR_ALL = 4;
}

enum InteractionAction {
	// 走线上逻辑，兼容原来逻辑，客户端判断应该触发什么
	Ignore = 0;
	// 端上走点赞动作流程
	DiggAction = 1;
	// 打开评论面板
	EnterCommentPannel = 2;
	// 跳转拍摄页
	EnterShoot = 3;
}

enum InteractionTagInterestLevel {
	VIDEO_TAG_INTEREST_LEVEL_UNDEFINED = 0;
	// FIND NOT INTEREST IN THIS TAG
	VIDEO_TAG_INTEREST_LEVEL_LOW = 1;
	// FIND PEOPLE IN IT ARE RELATED TO ONESELF
	VIDEO_TAG_INTEREST_LEVEL_HIGH = 2;
}

enum InteractionTypeEnum {
	NONE = 0;
	Deprecated1 = 1;
	VOTE = 3;
	COMMENT = 4;
	WIKI = 5;
	DONATION = 6;
	Deprecated7 = 7;
	MENTION = 8;
	HASH_TAG = 9;
	LIVE_COUNTDOWN = 10;
	AUTO_CAPTION = 11;
	// 12->15 are reserved - keep consistent with IDL file
	DUET_WITH_ME = 16;
	FORUM = 17;
	TEXT_STICKER = 18;
	SHARE_TO_STORY = 19;
	BURN_IN_CAPTION = 20;
	POI = 22;
	NATURE_CLASSIFICATION = 23;
	INTERACTION_LIVE_EVENT = 25;
	// comment post video's comment
	POST_COMMENT_COMMENT = 40;
	// comment post video's item
	POST_COMMENT_ITEM = 41;
	MUSIC_STICKER = 42;
	AI_SONG = 50;
	ADD_YOURS = 88;
	INTERACTIVE_EMOJI = 100;
	EMOJI_SLIDER = 101;
	LINK = 102;
	// The attribution link sticker for link sharing functionality
	ATTRIBUTION_LINK = 103;
}

enum MaskTypeEnum {
	REPORT = 1;
	DISLIKE = 2;
	// general mask
	GENERAL = 3;
	// photosensitie epilepsy
	PHOTOSENSITIVE = 4;
	// content classification mask
	CONTENT_CLASSIFICATION = 5;
}

// media type such as video or image
enum MediaInfoType {
	MediaInfoType_Unknown = 0;
	MediaInfoType_Video = 1;
	MediaInfoType_Image = 2;
}

enum MentionStickerScenario {
	MENTION_STICKER = 0;
	TEXT_STICKER = 1;
	STORY_TO_STORY = 2;
	POST_TO_STORY = 4;
}

// Item上宣推的tag类型
enum MusicPromoTagType {
	// 用户在wwa宣推投稿的tag
	PROMO_WWA_LABEL_TAG = 0;
}

enum MutualType {
	MUTUAL_TYPE_FRIEND = 1;
	MUTUAL_TYPE_FOLLOWED = 2;
	MUTUAL_TYPE_FOLLOW = 3;
	MUTUAL_TYPE_MUTUAL_CONNECTIONS = 4;
}

enum NowAvatarBadgeStatus {
	// no now avatar badge
	NoNowAvatarBadge = 0;
	// christmas new year avatar badge
	ChristmasNewYearAvatarBadge = 1;
	// now christmas new year avatar badge
	NonChristmasNewYeaAvatarBadge = 2;
}

enum NowBlurType {
	Ignore = 0;
	// 前置清晰
	FrontVisible = 1;
	// 高斯模糊，图片投稿为前置高斯，视频投稿为合成图高斯
	Gauss = 2;
}

enum NowCollectionType {
	NowCollectionTypeFlatted = 1;
	NowCollectionTypeAggregated = 2;
}

enum NowPostCameraType {
	// 默认双摄
	DefaultDual = 0;
	// 后置摄像头拍摄
	Rear = 1;
	// 前置摄像头拍摄
	Front = 2;
}

enum NowPostSource {
	NowPostSourceUnknown = 0;
	NowPostSourceFriends = 1;
	NowPostSourceFollowing = 2;
	NowPostSourcePopular = 3;
}

enum NowViewState {
	NowNotViewed = 0;
	NowViewedFuzzy = 1;
	NowViewedVisible = 2;
}

enum POIAnchorContentType {
	PoiAnchorContentType_FriendPostCNT = 0;
	PoiAnchorContentType_PeopleFavCNT = 1;
	PoiAnchorContentType_PeoplePostCNT = 2;
	PoiAnchorContentType_VideoCNT = 3;
	PoiAnchorContentType_Category = 4;
	PoiAnchorContentType_Guidance = 5;
}

enum POIContentExpType {
	PoiSubTagExpType_DirectExpand = 0;
	PoiSubTagExpType_TailExpand = 1;
	PoiSubTagExpType_HeadExpand = 2;
}

enum POIHideType {
	Suffix = 1000;
}

enum PermissionLevelEnum {
	UNKNOWN = 0;
	EVERYONE = 100;
	AUTHOR = 200;
	FRIENDS = 201;
	// custom：with specified user list
	CUSTOM = 202;
	FOLLOWER = 203;
	NO_ONE = 300;
}

enum PodcastFollowDisplay {
	// not show follow button on audio feed
	FollowNone = 0;
	// show follow button on audio feed
	Follow = 1;
}

enum PodcastSharkStatus {
	SHARK_STATUS_UNKNOWN = 0;
	SHARK_STATUS_PASS = 1;
	SHARK_STATUS_BLOCK_INVISIBLE = 2;
	SHARK_STATUS_BLOCK_REJECT = 3;
}

enum PodcastTnsStatus {
	TNS_STATUS_UNKNOWN = 0;
	TNS_STATUS_APPROVED = 1;
	TNS_STATUS_REJECTED = 2;
	TNS_STATUS_NOT_RECOMMENDED = 3;
}

enum PoiClaimStatusType {
	Draft = 0;
	Audit = 1;
	Pass = 2;
	Reject = 3;
	Expired = 4;
}

enum PoiReTagType {
	NoNeed = 0;
	General = 1;
}

// quick comment recommend level
enum QuickCommentRecLevelEnum {
	// Weak recommendation: allowed by the server, display is subject to client-side discretion
	QUICK_COMMENT_REC_LEVEL_ENUM_WEAK = 0;
	// Not recommended: not advised for client-side display
	QUICK_COMMENT_REC_LEVEL_ENUM_NOT_RECOMMENDED = 1;
	// Strong recommendation: expected to be displayed on the client-side
	QUICK_COMMENT_REC_LEVEL_ENUM_STRONG = 2;
}

enum RecreateLimit {
	// 可以
	CAN = 0;
	// 不可以
	CAN_NOT = 1;
	// 广告平台关闭该功能
	AD_CAN_NOT = 2;
}

enum RedirectPage {
	GeneralSearchPage = 0;
	EcomSearchPage = 1;
}

enum ShareStoryStatusEnum {
	// Unknown status
	UnknownShareStoryStatus = 0;
	// Has shared to story. Disable and do not show button
	DisableMentionTagShareStory = 1;
	// Has not shared to story. Enable and show button
	EnableMentionTagShareStory = 2;
}

// 组件展示策略
enum ShowStrategyEnum {
	// 普通组件
	ShowStrategyTypeDefault = 0;
	// 常驻组件，不参与规避规则
	ShowStrategyTypeSkipAvoidance = 1;
	// 在全局规避中必须展示
	ShowStrategyTypeMustShowInGlobalAvoidance = 2;
}

enum SoundExemptionStatus {
	NO_EXEMPTION = 0;
	SELF_PROMISE_EXEMPTION = 1;
}

// type of background that user chooses for story note item
enum StoryNoteBackgroundType {
	// unknown type
	BACKGROUNDTYPE_UNKNOWN = 0;
	// use avatar to create gradient color image
	BACKGROUNDTYPE_AVATAR_GRADIENT = 1;
	// create normal gradient color image
	BACKGROUNDTYPE_GENERAL_GRADIENT = 2;
	// use image directly
	BACKGROUNDTYPE_IMAGE = 3;
}

enum StoryType {
	NotUsed = 0;
	Default = 1;
	TextPost = 2;
	StoryTypeAIGCPost = 4;
}

// 促销文案类型
enum SuggestPromotionType {
	REDUCTION = 1;
	DISCOUNT = 2;
	TEXT = 3;
}

enum TCMReviewStatusEnum {
	InReviewing = 1;
	ReviewReject = 2;
	ReviewPassed = 3;
}

// tt2dsp anchor button type
enum TT2DspButtonType {
	// 0 by default, dont show
	DEFAULT = 0;
	// big button
	BIG = 1;
	// small button
	SMALL = 2;
	// show music title instead of add button
	MusicTitle = 3;
}

// tt2dsp 状态字段
enum TT2DspStatus {
	NoShowByUGCMusic = -4;
	// isPrivateAccount || isPrivateVideo || isSubOnlyVideo
	NoShowByPrivateVideo = -3;
	// video author is not verified artist and music volum < 21
	NoShowByMusicVolume = -2;
	// no show based on algo resp
	FypNoShowByAlgo = -1;
	DEFAULT = 0;
	// 在operation tool 配置白名单内
	SetByOp = 1;
	// fyp deliver button_type = 0 for trouble shooting & precheck
	FYPShowZeroButtonType = 2;
	// DSP2TT scene will always show tt2dsp anchor
	DSP2TTAlwaysShow = 3;
}

enum TTMProductType {
	Resso = 1;
	TTMusic = 2;
}

enum TTMUaSwitchStatus {
	TTMBetaSwitchOff = 1;
	TTMBetaSwitchOn = 2;
}

enum TaggingBaInvitationStatus {
	Unknown = 0;
	Pending = 1;
	Accept = 2;
}

enum TextTypeEnum {
	USER_TEXT = 0;
	CHALLENGE_TEXT = 1;
	COMMENT_TEXT = 2;
}

enum TrendingProductRecallType {
	NoEcomIntent = -2;
	HaveEcomIntent = -1;
	SAME = 1;
	SAME_AND_SIMILAR = 2;
	SIMILAR = 3;
	CATEGORY = 4;
}

enum UserNowStatus {
	NoVisibleNow = 0;
	HasVisibleNow = 1;
	ViewedVisibleNow = 2;
}

enum UserStoryStatus {
	NoVisibleStory = 0;
	HasVisibleStory = 1;
	AllViewedVisibleStory = 2;
}

enum VideoDistributeTypeEnum {
	// short video
	SHORT_VIDEO = 1;
	// long video
	LONG_VIDEO_DIRECT_PLAY = 2;
	// long video with short video
	LONG_VIDEO = 3;
}
