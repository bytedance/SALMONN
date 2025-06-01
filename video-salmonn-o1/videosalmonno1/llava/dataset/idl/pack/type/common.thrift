include "enum.thrift"
include "../content/content_standard_biz.thrift"


namespace go tiktok.pack_type.common
namespace py tiktok.pack_type.common

// Common Structs
struct UrlStruct {
    1: required string Uri,
    2: required list<string> UrlList,
    3: optional i64 DataSize,
    4: optional i32 Width,
    5: optional i32 Height,
    6: optional string UrlKey,
    7: optional string FileHash,
    8: optional string FileCs,
    9: optional string PlayerAccessKey,
    10: optional string FileId,
    11: optional string ImageMediaModelPbBase64,
    12: optional string ImageMediaModelJson,
    13: optional PreviewStruct Preview,
}

struct PreviewStruct {
    1: optional string Data,
    2: optional i32 PreviewType,
}

struct SuggestWordList {
    1: optional list<SuggestWordListStruct> SuggestWords, // suggest word list
}

struct SuggestWordListStruct {
    1: optional list<SuggestWordStruct> Words; // suggest word list
    2: optional UrlStruct IconUrl; // the icon url before the list
    3: optional string Scene; // the scene where the list appears
    4: optional string HintText; // the hint text before the list
    5: optional string ExtraInfo; // the extra info
    6: optional string QrecVirtualEnable;// virtual signal of suggest words for FE
}

struct SuggestWordStruct {
    1: optional string Word;
    2: optional string WordId;
    3: optional string Info;
}


struct TTECSuggestWordList {
    1: optional list<TTECSuggestWordListStruct> TTECSuggestWords, // TTEC suggest word list
}

struct TTECSuggestWordListStruct {
    1: optional list<TTECSuggestWordStruct> Words; // suggest word list
    2: optional UrlStruct IconUrl; // the icon url before the list
    3: optional string Scene; // the scene where the list appears
    4: optional string HintText; // the hint text before the list
    5: optional string ExtraInfo; // the extra info
    6: optional string QrecVirtualEnable;// virtual signal of suggest words for FE
    7: optional enum.RedirectPage RedirectPage;//跳转
}

struct TTECSuggestWordStruct {
    1: optional string Word;
    2: optional string WordId;
    3: optional string Info;
}


// Shared Structs
struct BillBoardStruct {
    1: required i32 Rank,
    2: optional i64 RankValue,
}

struct DubbingVideoStruct {
    1: required string Lang,
    2: required i64 LanguageId, // map<language, language_id> is maintained by vcloud. https://bytedance.feishu.cn/docs/doccnL3XBAihnFwawShre7pjgyh
    3: required string VoiceType = "mixed", // M/F/mixed
    4: optional UrlStruct PlayAddr,
    5: optional UrlStruct PlayAddrH265,
    6: optional UrlStruct PlayAddrByteVC1, // used to replace PlayAddrByteVC1
}

struct TtsAudioStruct {
    1: optional string Lang,            // Subtitle language code provided by VideoArch, e.g., "jpn-JP"
    2: optional i64 LanguageId,         // Subtitle language ID provided by VideoArch
    3: optional string VoiceType,       // Stringified enum value that represents a dubbing voice
    4: optional UrlStruct PlayAddr,     // Playback address for the audio track
    5: optional string VolumeInfo,      // Information for the client to adjust audio volume/equilibrium
    6: optional i32 BitRate,            // bit rate for this audio file, used for event tracking
    7: optional string LanguageCode,    // Dubbing language code to match with CLA client CaptionModel
}

struct OriginalSoundStruct {
    1: optional UrlStruct PlayAddr, // Playback address for the audio track
}

struct BitrateStruct {
    1: required string GearName,
    2: required i32 Bitrate,
    3: required i32 QualityType,
    4: optional UrlStruct PlayAddr,
    5: optional UrlStruct PlayAddrH265,
    6: required string CodecType,
    7: optional UrlStruct PlayAddrByteVC1, // used for replace PlayAddrByteVC1
    8: optional list<DubbingVideoStruct> DubInfos, // used for CLA dubbed videos
    9: optional string HdrType,  // HDR type : Normal-0, HLG-1, PQ-2
    10: optional string HdrBit,  // HDR bit depth
    11: optional string VideoExtra, // Video attachment information, json string, including information such as video first frame offset and VR playback parameters.
    12: optional string Format, // Video Format，mp4/dash
    13: optional i64 FPS, // frame rate
    14: optional i64 Bandwidth, // ABR bitrate
    15: optional string FidProfileLabels,
}

struct BitrateAudioStruct {
	// Audio metadata
	1: optional BitrateMetaStruct AudioMeta,
	// Audio quality
	2: optional i64 AudioQuality,
	// Audio extra information, json string
	3: optional string AudioExtra,
}

struct BitrateMetaStruct {
	// Play address
	1: optional BitrateUrlStruct UrlList,
	// Whether it is encrypted and the encryption method
	2: optional string EncodedType,
	// Media type
	3: optional string MediaType,
	// Watermark name
	4: optional string LogoType,
	// Resolution
	5: optional string Definition,
	// Quality level
	6: optional string Quality,
	// Quality description information
	7: optional string QualityDesc,
	// Format
	8: optional string Format,
	// Width
	9: optional i64 Width,
	// Height
	10: optional i64 Height,
	// Bit rate
	11: optional i64 Bitrate,
	// Encoder type
	12: optional string CodecType,
	// Size
	13: optional i64 Size,
	// Frame rate
	14: optional i64 FPS,
	// Unique identifier
	15: optional string FileId,
	// Hash unique identifier
	16: optional string FileHash,
	// Fields passed through to the VOD SDK
    17: optional string SubInfo,
    // ABR bitrate
    18: optional i64 Bandwidth,
    19: optional string FidProfileLabels,
}

struct BitrateUrlStruct {
    // Main playback url
    1: optional string MainUrl,
    // Backup playback url
    2: optional string BackupUrl,
    // Fallback url
    3: optional string FallbackUrl,
}

struct HotListStruct{
    1:required string ImageUrl,
    2:required string Schema,
    3:required string Title,
    4:required i32 Type,
    5:required string I18nTitle,
    6:optional string Header,
    7:optional string Footer,
    8:optional i32 PatternType,
}

struct FakeLandscapeVideoInfoStruct {
    1: optional double Top,
    2: optional double Bottom,
    3: optional double Left,
    4: optional double Right,
    5: optional i32 FakeLandscapeVideoType
}

struct PillarBoxVideoInfoStruct {
    1: optional double Top,
    2: optional double Bottom,
    3: optional double Left,
    4: optional double Right,
    5: optional string Version
}

struct DontShareStruct{
    1:required i32 VideoHideSearch,
    2:required i32 DontShare,
}

struct LabelStruct {
    1: required byte Type,
    2: required string Text,
    3: optional UrlStruct Url,
    4: optional string Color,
    5: optional string ColorText,
    6: optional string RefUrl,
    7: optional string TextKey, // starling key for text
    8: optional i64 RecommendType, // relation label recommend type, from recommend service
}

struct MarkerStruct {
    1: optional bool CanComment,
    2: optional bool CanCommentForAd,
    3: optional i32 CanCommentStatus,
    4: optional i32 CanDownloadStatus,
    5: optional i32 CanDuetStatus,
    6: optional i32 CanReactStatus,
    7: optional bool HasLifeStory,
    8: optional bool HasOrders,
    9: optional bool IsAdFake,
    10: optional bool IsAds,
    11: optional bool IsCanceled,
    12: optional bool IsCommerceChallenge,
    13: optional bool IsCollected,
    14: optional bool IsCopyCat,
    15: optional bool IsDelete,
    16: optional bool IsDouPlus,
    17: optional bool IsFantasy,
    18: optional bool IsFriend,
    19: optional bool IsInstituteMediaVip,
    20: optional bool IsHashTag,
    21: optional bool IsMyself,
    22: optional bool IsNotRecommend,
    23: optional bool IsOnlyOwnerUse,
    24: optional bool IsOriginal,  //is original musician music
    25: optional bool IsPgcshow,
    26: optional bool IsPrExempted,
    27: optional bool IsPreventDownload,
    28: optional bool IsRealChallenge,
    29: optional bool IsRelieve,
    30: optional bool IsRedirect,
    31: optional bool IsRestricted,
    32: optional bool IsSelfSee,
    33: optional bool IsSecretAccount,
    34: optional bool IsSpecialVr,
    35: optional bool IsTop,
    36: optional bool WithCommerceEntry,
    37: optional bool WithFusionShopEntry,
    38: optional bool WithQuickShop,
    39: optional bool WithShopEntry,
    40: optional bool WithStoryPrivilege,
    41: optional i32 WithoutWatermarkStatus, // 0: download high quality video without watermark;1: ad owner assign
    42: optional bool IsFromAd,
    43: optional i32 EffectDesignerStatus, // default null, 1 stands need sign, 0 stands no
    44: optional bool Deprecated44,
    45: optional bool CanCommentForGr,
    46: optional bool CanShareForGr,
    47: optional bool CanForwardForGr,
    48: optional bool IsCommerceMusic,
    49: optional bool IsOriginalSound,
    50: optional bool Deprecated50,
    51: optional enum.MusicVideoStatusEnum CopyrightVideoStatus,   //The influence of music copyright on video
    52: optional bool IsSelfSeeExcptOwner,   //Video self seeing users have no perception
    53: optional i32 AdAuthStatus, // Whether it is authorized to advertisers, 0 is not authorized; 1 is not authorized; 2 is authorized
    54: optional bool IsMute, // Is the video muted due to music problems
    55: optional bool MuteShare, //Whether to share in silence
    56: optional bool AdverHookup, // Account is related to advertisers.
    57: optional bool IsLooseNotRecommend,
    58: optional bool IsStrictNotRecommend,
    59: optional bool IsMuteAndProfileSee,    // The video is muted and visible only on the personal page
    60: optional bool IsDmvAutoShow,
    61: optional bool AuctionAdInvited, // Has the video ever been invited to bid for MT advertising
    62: optional bool Deprecated62,
    63: optional bool IsPgc, // is pgc music
    64: optional bool IsLiveReply, // is total live replay
    65: optional bool WithCommentFilterWords,
    66: optional bool WithDashboard,
    67: optional bool AdvPromotable, // Are advertisers allowed to promote
    68: optional bool isCollectedMix,
    69: optional i32  CanStitchStatus, // Stick setting of video dimension
    70: optional string ReviewTooStrictTestLabel, // Effect of too strict audit on author experience
    71: optional bool CanBackground, // Can it be played in the background
    72: optional i32 AdBanMask, // ban some action by commerce reason
    73: optional bool AllowGift, // true if the user can send gift to the aweme creator
    74: optional i32 QaStatus, // Q&A feature status, 0 - shut down, 1 - turn on
    75: optional i32 ItemMuteDownloadStatus  // 2: item mute download status
    76: optional enum.SoundExemptionStatus SoundExemption // whether sound of video is exempted
    77: optional bool PlayListBlocked // whether the current item is not allowed to be added to playlists
    78: optional string PartN // position info for long split videos
    79: optional enum.UserStoryStatus StoryStatus,  // author story status
    80: optional enum.MusicCommercialRightType CommercialRightType, // commercial rights of music: 1 --> non-commercial music; 2 --> commercial music; 3 --> commercial music with private labels
    81: optional UserNowPackStruct UserNowPackInfo, // user now related info https://bytedance.feishu.cn/docx/doxcnnQIexO7A8NghhR1iVjUvbd
    82: optional i32 PinnedVideoStatus, // Aritst Pinned Video PRD: https://bytedance.sg.feishu.cn/docx/doxlg6lmWgCifPtUHto0dgdTCGd
    83: optional bool IneligibleForFeed,  // user IFF status, read value from user.base_info TnSRecommendStatus.IneligibleForFeed
    84: optional bool IsOpenTestAccount, // TTS OPEN test account
    85: optional bool IsSubOnlyVideo, // identify current video is sub only video or not
    86: optional bool IsSubscriber, // identify current user is subscribe to video poster or not
    87: optional bool MuteAndProfileSeeOnlyByLongVideo, // The long video use this music is muted and visible only on the personal page
    88: optional bool IsECommerce, // judge is ecommerce video
    89: optional i64 BrandedContentType, // // extra.branded_content_type
    90: optional bool IsOnlyMusicContractPunishment // identify current music only punish by music contract
    91: optional bool PrivateExempt // identify current music can be exempt by special user
    92: optional bool IsNotRecommendBySafety // whether the current music is not recommend by safety
    93: optional i32 IsMutePost // mute user post, only used in pack user
    94: optional i32 SubOnlyVideoEnrollStatus // creator's sub only video enroll status, only used in pack user
    95: optional bool IsBaVideoMute // ba video need mute, only used in MusicGroup
    96: optional UserAllStoryMetadataStruct UserAllStoryMetadata // user story metadata related info, only used in pack user https://bytedance.sg.larkoffice.com/docx/Ir5zdz33ZoT4hkxxKPYlAu9WgV8
    97: optional i32 IsMuteStory // mute user story, only used in UserGroup
    98: optional i32 IsMuteLives // mute user lives, only used in UserGroup
    99: optional i32 IsMuteNonStoryPost // mute user non-story post, only used in UserGroup
}

// Masked information
struct MaskStruct {
    1: required enum.MaskTypeEnum MaskType,  // Mask type
    2: optional i32 Status,                       // Mask status
    3: optional string Title,                     // Mask title
    4: optional string Content,                   // Mask content
    5: optional string CancelMaskLabel,           // cancel Mask label
    6: optional MaskExtraModule PopWindow,
    7: optional MaskExtraModule BirthdayEditModule, // 生日编辑入口
    8: optional MaskExtraModule PolicyModule,
    99: optional string Extra,
}

struct MaskExtraModule {
    1: required i32 ModuleType, // 模块类型： 1.pop window 2. birthday edit module 3. policy module
    2: optional string BtnText, // 入口按钮文案
    3: optional string Url, // 跳转链接（如有）
    4: optional MaskPopWindow PopWindow, // 弹窗结构
}

struct MaskPopWindow {
    1: optional string PopTitle,
    2: optional list<MaskPopText> PopContent,
}

struct MaskPopText {
	1: required TextWithInlineLink Text,
	2: optional bool IsListItem,
}

struct TextWithInlineLink {
	1: required string Text, // ATTENTION!! Only support placeholder style "%1$s, %2$s"
	2: optional list<InlineLink> Links,
}

struct InlineLink {
    1: required string PlaceHolder,
    2: required string Text,
    3: optional string Url,
}

// Activity information
struct ActivityStruct {
    1: optional string ActivityId,                // ActivityID
    2: optional i64 ShowDelayTime,                // Delay trigger time
    3: optional string SchemaUrl,
    4: optional string ContentText,               // ContentText
    5: optional string ContentColor,              // ContentColor
    6: optional string ContentSize,               // ContentSize
    7: optional string ButtonLabel,               // Button content
    8: optional string ButtonColor,               // Button content color
    99: optional string Extra,
}


struct ShareStruct {
    1: required string Desc,                 // share description
    2: required string Title,                // share title
    3: required string Url,                  // share url
    4: required string ContentDesc,          // share content desc
    5: optional UrlStruct ImageUrl,          // share image url
    6: optional string LinkDesc,             // share link desc
    7: optional UrlStruct QrcodeUrl,         // share personal link qr code
    8: optional i32 PersistStatus,           // 0 valid for 6 months, 1 Long term effectiveness
    9: optional string Quote,                // fb ins share content
    10: optional string SignatureDesc,       // jp share content
    11: optional string SignatureUrl,        // jp share url
    12: optional string WhatsappDesc,        // whatsapp share content
}

struct TextExtraStruct {
    1: required enum.TextTypeEnum Type,
    2: required i64 Start,
    3: required i64 End,
    4: optional i64 HashtagId,
    5: optional string HashtagName,
    6: optional string UserId,
    7: optional bool IsCommerceHashtag,
    8: optional string SecretUserId,
    9: optional string AwemeId,
    10: optional i64 StickerId, // image StickerId
    11: optional UrlStruct StickerUrl, // image sticker url
    12: optional i32 StickerSource, // image sticker source
    13: optional i32 SubType,
    14: optional i64 QuestionId, // forum question id
    15: optional i32 LineIdx, // index for caption lines
    16: optional string TagId, // the id of text extra, used in markup.
}

struct TimeRange {
    1: required double Start,
    2: required double End,
}

// Base Structs
struct ACLCommonStruct {
    1: optional i32 Code,
    2: optional i32 ShowType,
    3: optional string ToastMsg,
    4: optional string Extra,
    5: optional i32 Transcode,
    6: optional bool Mute,
    7: optional string PopupMsg,
    8: optional string PlatformId,
}

struct ActivityTrilateralCooperationStruct {
    1: optional string Desc
    2: optional string Title
    3: optional string JumpUrl
    4: optional string IconUrl
    5: optional bool IsTask
    6: optional i32 SwitchType
    7: optional string EntranceUrl
}

struct ActivityCommerceStruct {
    1: required enum.CommerceActivityTypeEnum ActType, // red envelope type 1: gesture red envelope ，2: KOL pendant
    2: required string JumpOpenUrl,                         // openurl
    3: required string JumpWebUrl,                          // H5url
    4: optional string Title,
    5: optional UrlStruct Image,
    6: optional i64 StartTime,
    7: optional i64 EndTime,
    8: optional list<TimeRange> TimeRanges,
    9: optional string TrackUrl,       // Third party monitoring url
    10: optional string ClickTrackUrl, // Third party click monitoring url
}

struct AddressStruct {
    1: required string Address,
    2: required string City,
    3: required string CityCode,
    4: required string Country,
    5: required string CountryCode,
    6: required string District,
    7: required string Province,
    8: required string SimpleAddr,
    9: optional string CityTrans,
    10: optional string CountryTrans,
    11: optional string ProvinceTrans,
    12: optional string AdCodeV2,
    13: optional string CityCodeV2,
}

struct CategoryStruct {
    1: required string Code,
    2: required string Name,
}

struct ConfigDataItemLikeEggStruct {
    1: optional string MaterialUrl,
    2: optional string FileType,
}

struct ConfigDataStickerPendantStruct {
    1: optional i32 StickerType,
    2: optional string Link,
    3: optional string Title,
    4: optional string StickerId,
    5: optional UrlStruct IconUrl,
    6: optional string OpenUrl,
}

struct CommerceConfigStruct {
    1: required string Id,
    2: required i32 Type,
    3: required string Data,
    4: optional i32 TargetType,
    5: optional string TargetId,
    6: optional ConfigDataItemLikeEggStruct ItemLikeEgg,
    7: optional ConfigDataStickerPendantStruct StickerPendant,
}

struct CommentImageStruct {
    1: optional UrlStruct OriginUrl
    2: optional UrlStruct CropUrl

}

struct CommentStruct {
    1: required i64 Id,
    2: required i64 ItemId,
    3: required i64 CreateTime,
    4: required i64 DiggCount,
    5: required string Extra,
    6: required i64 ReplyId,
    7: required i16 Status,
    8: required bool UserDigged,
    9: required string Text,
    10: required list<TextExtraStruct> TextExtraInfos,
    11: required i64 UserID,
    12: required i64 GroupID,
    13: required i64 ReplyToCommentId,
    14: optional i32 Level,
    15: optional bool IsReply,
    16: optional i64 ReplyToUserId,
    17: optional i32 LabelType,
    18: optional string LabelText,
    19: optional i32 CommentType,
    20: optional i64 ParentId,
    21: optional CommentStruct ReplyToComment,
    22: optional string ForwardId,
    23: optional CommentStruct Level1Comment,
    24: optional i64 AliasGroup,
    25: optional i64 AppId,
    26: optional bool IsFavorited, // not in use
    27: optional bool CollectStat, // whether or not comment has been collected
    28: optional list<CommentImageStruct> CommentImage, // comment photos
}

struct Deprecated1Struct {
    1: required string Id,
    2: required string Deprecated2,
    3: required i32 ExpandType,
    4: required i32 IconType,
    5: required i32 ItemCount,
    6: required double Deprecated6,
    7: required double Deprecated7,
    8: required string Name,
    9: required string TypeCode,
    10: required i32 UserCount,
    11: required ShareStruct ShareInfo,

    12: optional AddressStruct AddressInfo,
    13: optional UrlStruct CoverHd,
    14: optional UrlStruct CoverLarge,
    15: optional UrlStruct CoverMedium,
    16: optional UrlStruct CoverThumb,
    17: optional UrlStruct Deprecated17,
    18: optional UrlStruct IconOnInfo,
    19: optional UrlStruct Deprecated19,
    20: optional i32 ShowType,
    21: optional string Subtitle,
    22: optional string SpSource,
    23: optional i32 SubtitleType,
    24: optional string Voucher,
    25: optional list<string> VoucherReleaseAreas,
    26: optional list<CategoryStruct> FrontendCategoryInfos,
    27: optional CategoryStruct BackendCategoryInfo,
    28: optional i64 ClaimerUserID,
    29: optional string ClaimerNickname,
    30: optional string ClaimerAvatarUrl,
    31: optional string ThirdId,
    32: required i64 ViewCount,
    33: optional string ItemTag,
    34: optional i64 CollectCount,            // Number of collections
    35: optional list<UrlStruct> HeadImages,
    36: optional bool Deprecated36,
    37: optional ItemLabelStruct Deprecated37;
    38: optional string Deprecated38;
}

struct PromotionVisitorStruct {
    1: required list<UrlStruct> Avatars, // Picture of visitors
    2: required i32 Count,               // Total views
}

struct SearchStruct {
    1: required string Sentence,
    2: optional string ChallengeId,
    3: optional i64 HotValue,
    4: optional string SearchWord,
    5: optional i32 Rank,
    6: optional list<string> RestrictedRegion  // Restricted search in this area
    7: optional bool HasVisualSearchEntry,     // Visual search portal (video on or off)
    8: optional string GroupId,                // 热搜词group_id
    9: optional bool NeedVisualSearchEntry,    // visual search portal (whether the user opens it or not)
    10: optional i32 Label,
    11: optional i32 PatternType,              // Hot bar style
    12: optional VisualSearchEntryStruct VisualSearchEntry  // Entry style of related video in feed
}

// @docs: https://bytedance.feishu.cn/docs/doccncSf33uLrilfPYvgpWnaImb#
struct VisualSearchEntryStruct {
	1: optional string Title
	2: optional string SubTitle
	3: optional string TextColor
	4: optional UrlStruct IconUrl
}


struct Deprecated2Struct {
    1: optional enum.Deprecated1TypeEnum Deprecated1, //discard
    2: optional enum.Deprecated1TypeEnum Deprecated2,
}

struct StickerStruct {
    1: required string Id,
    2: required UrlStruct IconUrl,
    3: required string Link,
    4: required string Title,
    5: required i32 Type,
    6: optional string Name,
    7: optional string DesignerId,
    8: optional string DesignerEncryptedId,
    9: optional i64 UserCount,
    10: optional list<string> Tags,
    11: optional string OpenUrl,
}

struct NearbyStruct {
    1: optional string EventTrack,
}

struct XSpaceStruct {
    1: required i64 RoomId,       // XS room ID
    2: required i64 Duration,     // XS connection time, in seconds
    3: required i64 XSTime,       // XS connection date, timestamp
    4: required string XSTitle    // title of each notice
    5: required string XSContent  // details of each notice
}

struct VideoControlStruct {
    1: optional bool AllowDownload,        // can I download it
    2: optional bool AllowDuet,            // match
    3: optional bool AllowReact,           // is it possible to catch the camera
    4: optional i32 DraftProgressBar,      // 0 can not be dragged, 1 progress bar can be dragged
    5: optional i32 ShareType,             // 0 not allowed to share 1 share download 2 share QR code
    6: optional i32 ShowProgressBar,       // 0 do not display progress bar, 1 display progress bar
    7: optional i32 PreventDownloadType,   // unable to download reason type 0 watermark transcoding not completed 1 audit failed
    8: optional bool AllowDynamicWallpaper,// true allows dynamic wallpaper, false does not allow dynamic wallpaper
    9: optional i32 TimerStatus,           // 1: timing in progress, 0: timing end
    10: optional bool AllowUseMusic,       // if you can use music, go to the music details page
    11: optional bool HideDownload,        // hide download button
    12: optional bool HideDouplus,         // hide the Dou + button
    13: optional bool AllowStitch,         // can the current video be stuck by others
}

struct VideoStruct {
    1: required string Id,
    2: required i32 Height,                                // height
    3: required i32 Width,                                 // width
    4: required i32 Duration,                              // Video duration
    5: required string Ratio,                              // Resolution (default, 360p, 540p, 720p)
    6: required bool HasWatermark,                         // Does the download video contain dynamic watermark
    7: required UrlStruct Cover,                           // Cover address
    8: required UrlStruct OriginCover,                     // Big picture cover address
    9: optional list<BitrateStruct> BitrateInfos,          // Bitrate configuration
    10: optional UrlStruct DynamicCover,                   // Dynamic cover
    11: optional UrlStruct PlayAddrH264,                   // H264 play address
    12: optional UrlStruct PlayAddr,                       // Play address
    13: optional UrlStruct DownloadAddr,                   // Download address
    14: optional UrlStruct PlayAddrLowbr,                  // Low bit rate playback address
    15: optional UrlStruct NewDownloadAddr,                // Download address of star video
    16: optional UrlStruct DownloadSuffixLogoAddr,          // Video with end watermark
    17: optional UrlStruct PlayAddrH265,                   // play address
    18: optional string CodecType,                         // Coding format
    19: optional UrlStruct UIAlikeDownloadAddr,            // Download address of similar UI watermark
    20: optional UrlStruct CaptionDownloadAddr,            // Caption watermark download address
    21: optional i32 ContentType,                          // Video content type
    22: optional VideoControlStruct ControlInfo,           // Control fields for Distributed Video
    23: optional i64 CdnUrlExpired,                        // cdl_ URL expiration time, UTC time, independent of time zone, unit specific to seconds
    24: optional i32 IsLongVideo,                          // Is it a long video (1-15min)
    25: optional string VideoModel,                        // Play address
    26: optional UrlStruct AnimatedCover,                  // The new 6-Frame dynamic cover reduces client memory consumption compared with the old 9-frame dynamic cover cover cover.
    27: optional bool NeedSetToken,                        // When accessing the video address, do you need to provide identity information
    28: optional UrlStruct AiCover,                        // Video cover extracted from video content by AI Lab
    29: optional double CoverTsp,                          // Whether the user selects the cover manually, > = 0 indicates the preferred cover, and the value represents the time
    30: optional map<string, UrlStruct> MiscDownloadAddrs, // Scenario and other customized download address, key includes snapchat, Lite, suffix_ scene...
    31: optional bool IsCallBack,                          // Complete transcoding callback
    32: optional UrlStruct PlayAddrByteVC1,                // used for replace PlayAddrByteVC1
    33: optional list<BigThumb> BigThumbs,                 // Sprite chart
    34: optional string Meta,                              // Video Meta Info, A map json string: map<string, string>
    35: optional UrlStruct AiDynamicCover,                 // Video dynamic cover extracted from video content by AI Lab
    36: optional UrlStruct AiDynamicCoverBak,              // Video dynamic cover extracted from video content by AI Lab (compressed)
    37: optional ClaStruct ClaInfo,                        // cross languge accessibility(CLA) info
    38: optional i32 SourceHDRType,                        // whether HDR video from the viewer: 0,1,-1 (-1 means viewer == creator, 0 not HDR, 1 HDR)
    39: optional string MiscDownloadAddrsStr,              // string of MiscDownloadAddrs
    40: optional list<BitrateAudioStruct> BitrateAudioInfos, // audio bitrate
    41: optional string Format,                            // video format, dash or mp4
    42: optional FakeLandscapeVideoInfoStruct FakeLandscapeVideoInfo, // struct to hold fake landscape video info
    43: optional PillarBoxVideoInfoStruct PillarBoxVideoInfo,
    44: optional string DidProfileLabels,                   // Device-Level portrait
    45: optional string VidProfileLabels,                   // Video-Level portrait
}

struct AudioStruct {
    1: optional i64 CdnUrlExpired,              // CDN URL expiration time in UTC with the smallest unit in second
    2: optional list<TtsAudioStruct> TtsInfos,  // List of TTS audio track information
    3: optional list<OriginalSoundStruct> OriginalSoundInfos, // list of original sounds information
}

struct BigThumb {
    1: i64 ImgNum
    2: string ImgURI
    3: string ImgURL
    4: i64 ImgXSize
    5: i64 ImgYSize
    6: i64 ImgXLen
    7: i64 ImgYLen
    8: double Duration
    9: string Fext
    10: double Interval
    11: list<string> ImgURIs
    12: list<string> ImgURLs
}

struct VoteOptionStruct {
    1: required i64 Id,
    2: required i64 Count,
    3: required string Text,
}

struct VoteStruct {
    1: required i64 Id,
    2: required i64 CreateTime,
    3: required i64 CreatorId,
    4: required i64 EndTime,
    5: required list<VoteOptionStruct> OptionInfos,
    6: required string Question,
    7: required i64 RefId,
    8: required i64 RefType,
    9: optional i64 SelectedOptionId,
    10: required i64 StartTime,
    11: required i32 Status,
    12: required i64 UpdateTime,
}

struct MentionStickerStruct {
    1: required string UserName,
    2: optional string SecUid,
    3: optional string UserId,
    4: optional string Nickname,
    5: optional UrlStruct UserAvatarUrl,
    6: optional enum.MentionStickerScenario Scenario,
}

struct HashTagStickerStruct {
    1: required string HashTagName,
    2: optional i64 HashTagId,
    3: optional i32 Status, // 0: unavailable 1: available
}

struct LiveCountdownStickerStruct {
    1: required string Title,
    2: required i64 EndTime,
    3: optional i64 SubscribedCount,
    4: optional bool IsSubscribed,
    5: optional string TextTobeSubscribed,
    6: optional string TextAlreadySubscribed,
    7: optional string TextAlreadyExpired
}

struct AutoCaptionType {
    1: optional string Language,
    2: optional UrlStruct Url,
}

struct AutoVideoCaptionStickerStruct {
    1: optional enum.AutoCaptionLocationType Location,
    2: optional list<AutoCaptionType> AutoCaptions,
    3: optional AutoCaptionPosition Position,
    4: optional AutoCaptionAppearance Appearance,
}

struct AutoCaptionPosition {
    1: optional list<double> Vertical,
    2: optional double Horizontal,
}

struct AutoCaptionAppearance {
    1: optional string BackgroundColour, //bg_color
    2: optional double BackgroundCornerRadius, //bg_corner_radius
    3: optional i32 TextLabelAlignment,
    4: optional list<i32> TextLabelInsets,
    5: optional i32 CaptionTextSize,
    7: optional string CaptionTextColor,
    8: optional double CaptionTextStrokeWidth,
    9: optional string CaptionTextStrokeColor,
    10: optional bool ShouldShowCaptionIcon,
    11: optional bool ShouldShowInstructionText,
    12: optional i32 InstructionTextSize,
    13: optional string InstructionTextColor,
    14: optional double InstructionTextStrokeWidth,
    15: optional string InstructionTextStrokeColor,
    16: optional i32 ExpansionDirection,
    17: optional TextLabelInsetInfoStruct TextLabelInsetInfo,
    18: optional CaptionControlInfoStruct CaptionControlInfo,
}

struct DuetWithMeStickerStruct {
    1: optional bool micPermissionOn, // auto-turn on/off mic for duet
    2: optional string stickerContent, //content of duet with me sticker like "Duet with me"
}

struct OriginalSharedVideoInfoStruct {
    1: optional string originalAuthorId, // authorId of the original video
    2: optional string OriginalAuthorName, // author name of the original video
    3: optional string originalItemId, // awemeId of the original video
    4: optional string originalSecAuthorId, // encrypted authorId
    5: optional i32 originalIndex, // index of the original post
}

struct PoiStickerInfoStruct {
    1: optional string PoiId, // poi_id of the sticker
}

struct NatureClassificationStickerStruct {
    1: optional i64 SpeciesId,
    2: optional string SpeciesName,
    3: optional i64 GenusId,
    4: optional string GenusName,
    5: optional string RedirectionUrl,
}

// Special Structs
struct InteractionStruct {
    1: required enum.InteractionTypeEnum Type,
    2: required i32 Index,
    3: optional string Attr,
    4: optional Deprecated1Struct Deprecated4,
    5: optional string TrackInfo,
    6: optional VoteStruct VoteInfo,
    7: optional string  TextInfo,
    8: optional MentionStickerStruct MentionInfo,
    9: optional HashTagStickerStruct HashTagInfo,
    10: optional LiveCountdownStickerStruct LiveCountdownInfo,
    11: optional AutoVideoCaptionStickerStruct AutoCaptionInfo,
    12: optional DuetWithMeStickerStruct DuetWithMe,
    // 13: place holder
    14: optional QuestionStickerStruct QuestionInfo,
    15: optional TextStickerStruct TextStickerInfo,
    17: optional OriginalSharedVideoInfoStruct VideoShareInfo,
    18: optional PoiStickerInfoStruct PoiInfo,
    19: optional NatureClassificationStickerStruct NatureClassificationInfo,
    20: optional bool IsNonGlobal,
    21: optional i32 MaterialIndex,
    22: optional AddYoursStickerStruct AddYoursSticker,
}

// Challenge Structs
struct ChallengeBannerStruct {
    1: optional UrlStruct Icon,
    2: optional string WebUrl,
    3: optional string OpenUrl
}

struct QuestionStickerStruct {
    1: optional i64 QuestionId,
    2: optional i64 UserId,
    3: optional i64 ItemId,
    4: optional string Content,
    5: optional string Username,
    6: optional UrlStruct UserAvatar,
    7: optional string SecUid,
    8: optional ShareStruct InviteShareInfo,
    9: optional string Extra, // extra info stored as json string
    10: optional string CategoryMeta, // json str map<category_x, category_name> question, used for event tracking; example: '{category_1:foo,category_2:bar}'
}

struct TextStickerStruct {
    1: optional i32 TextSize,
    2: optional string TextColor,
    3: optional string BgColor,
    4: optional string TextLanguage,
    5: optional double SourceWidth,
    6: optional double SourceHeight,
    7: optional i32 Alignment,
}

struct ChallengeCommerceStruct {
    1: optional UrlStruct BackGroundImageUrl,
    2: optional string DisclaimerTitle,
    3: optional string DisclaimerContent,
    4: optional string LinkAction,
    5: optional string LinkText,
    6: optional string StickerId,
}

struct ChallengeI18nStruct {
    1: required string Desc,
    2: required string Language,
    3: required string Name,
    4: optional string DisclaimerContent,
}

struct ChallengeMaterial{    // source material
    1: required i32 SubType, // 1: Movie 2: Stars
    2: required i64 BindId,  // Can search through this ID
    3: optional i32 Type,    // Challenge type 1: film 2: TV series 3: variety show 4: Stars 5: others
}

struct ChallengeMediaSourceButtonStruct {
    1: required UrlStruct Icon,     // Icon in button
    2: required string Name,        // Copy on button
    3: required i32 ButtonType,     // 1-native playback 2-applet 3-collection
    4: required string Schema,      // Jump agreement
    5: optional i64 Eid,            // eid
    6: optional i64 Aid,            // aid
}

struct ChallengeOtherStruct {
    1: required i32 SubType,
    2: required i32 CommerceType,           // MT commercialization Mini challenge
    3: optional string HashTagProfile,      // Topic head map of operation configuration
    4: optional string AutoHashTagProfile,  // Automatically set topic header
    5: optional bool HasMedia,              // Is the topic bound to movie cards
    6: optional string BackgroundImage,     // Topic background map of operation configuration
    7: optional string Extra,               // Extra field of transparent transmission
}

struct ChallengeRelatedMediaSourceStruct {
    1: required string Title,                               // title
    2: required string ReadMore,                            // See more copywriting
    3: required string ReadMoreUrl,                         // See more links
    4: required i32 MediaType,                              // 1 TV drama 2 animation 3 film 4 variety show 5 news 6 music 7 sports 8 documentary 9 children 10 games 11 live 12 short content collection
    5: required string MediaName,                           // Name of film and television ensemble
    6: required UrlStruct Cover,                            // cover
    7: optional string ReleaseDate,                         // Release date
    8: optional i32 Duration,                               // Duration in minutes
    9: optional i32 SeqsCount,                              // Number of TV series or variety shows
    10: optional list<string> Classification,               // Plot classification (family, comedy)
    11: optional double Rating,                             // score
    12: required i32 RatingStatus,                          // Rating status
    13: required string MediaTag,                           // Tag
    14: optional ChallengeMediaSourceButtonStruct button,   // Button area
    15: optional i32 MediaSrcType,                          // 1 - film source 2 - no film source, no ticket link 3 - no film source, with ticket link
    16: optional string MediaDesc,                          // Introduction to the film
    17: required i64 CompassId,                             // Media ID
    18: optional string EpisodeInfoRawData                  // Episode information

}

struct ChallengeStatisticsStruct {
    1: required i32 UseCount,
    2: required i64 ViewCount,
    3: optional i64 UseCountI64,
}

// Item Structs
struct ItemACLStruct {
    1: optional ACLCommonStruct DownloadGeneral,
    2: optional map<string, ACLCommonStruct> DownloadOther,
    3: optional i32 ShareListStatus,
    4: optional ACLCommonStruct ShareGeneral,
    5: optional list<ACLCommonStruct> PlatformList,
    6: optional ACLCommonStruct ShareThirdPlatform,
}

struct ItemActivityStruct {
    1: optional ActivityCommerceStruct ActivityPendantInfo,
    2: optional ActivityCommerceStruct GestureRedPacketInfo,
    3: optional ActivityTrilateralCooperationStruct TrilateralCooperationInfo,
}

struct ItemAdStruct {
    1: optional list<CommerceConfigStruct> CommerceConfigInfos,
    2: optional string LinkRawData,
    3: optional string LinkSendData,
    4: optional i32 LinkType,
    5: optional string Schedule, // Advertising plan to be launched, separated by '\ n'
    6: optional string MarioRawData, // Mario returned raw_data
    7: optional string LiftMarkValue, // aka study_id
    8: optional string AdTitle,
    9: optional string AdAvatarUri,
    10:optional i32 DarkPostSource,
    11:optional i32 DarkPostStatus,
    12:optional i32 MissionItemStatus,
    13:optional i64 MissionId,
    14:optional enum.DiggShowScene DiggShowScene,
}

struct AnchorStrategyStruct {
    1: optional bool SupportSingleAnchor;
    2: optional bool SupportMultiAnchors;
    3: optional bool RemoveMvInfo;
}

struct ItemAnchorStruct {
    1: required enum.AnchorType Type
    2: optional string Keyword
    3: optional string Lang
    4: optional enum.AnchorState State
    5: optional string Url
    6: optional i64 Id
    7: optional string Extra
}

struct ItemAnchorMTStruct {
    1: optional string Id,
    2: required enum.AnchorType Type,
    3: optional string Keyword,
    4: optional string Url,
    5: optional UrlStruct Icon,
    6: optional string Schema,
    7: optional string Language,
    8: optional string Extra,
    9: optional string DeepLink,
    10: optional string UniversalLink,
    11: optional enum.AnchorGeneralType GeneralType,
    12: optional string LogExtra,
    13: optional string Description,
    14: optional UrlStruct Thumbnail,
    15: optional list<AnchorActionStruct> Actions,
    16: optional map<string, string> ExtraInfo,
    17: optional bool IsShootingAllow,
    18: optional string ComponentKey,
    19: optional string Caption, // feed 上的第二行文案
}

struct AnchorActionStruct {
    1: optional UrlStruct Icon,
    2: optional string Schema,
    3: optional enum.AnchorActionType action_type,
}

struct ItemAnchorStructV2 {
    1: required string Id,
    2: required i32 Type,
    3: optional string Title,
    4: optional string OpenUrl, // Jump to native page or out of end
    5: optional string WebUrl,  // Jump inside landing page
    6: optional string MpUrl,   // Jump applet
    7: optional UrlStruct Icon,
    8: optional string TitleTag, // Anchor type name
    99: optional string Extra,
    100: optional string LogExtra,
}

struct ItemCaptionAnchorStruct{
    1: optional string Keyword;
    2: optional string Link,
    3: optional UrlStruct Icon,
}

struct ItemCaptionStruct{
    1: required string Keyword,
    2: required string Link,
}

struct ItemCloudGameEntranceStruct {
    1: required string ButtonColor,
    2: required string ButtonTitle,
    3: optional string StickerInfoUrl,
    4: optional string StickerTitle,
    5: optional i32 ShowStickerTime,
}

struct ItemCloudGameStruct{
    1: required string CloudGameId,
    2: required ItemCloudGameEntranceStruct EntranceInfo,
    3: optional string DownloadUrl,
    4: optional string Extra,
}

// common label
struct ItemCommonLabelStruct {
    1: optional ItemLabelStruct Deprecated1;
}

struct ItemGameStruct {
    1: required enum.GameTypeEnum Type,
    2: required i32 Score,
}

struct ItemImageStruct {
    1: required string Id,
    2: required i32 Height,
    3: required i32 Width,
    4: required UrlStruct Large,
    5: required UrlStruct Thumb,
}

struct ItemLabelStruct {
    1: optional i32 Type;
    2: optional string Text;
}

// Search caption keyword matching information
struct ItemLinkMatchStruct{
    1: required i32 TotalLimit,
    2: required i32 QueryLimit = 2;
    3: optional list<ItemMatchStruct> MatchInfo,
    4: optional ItemCaptionStruct CaptionInfo,
    5: optional ItemCaptionAnchorStruct CaptionAnchor,
}

struct ItemMatchStruct{
    1: required string Query,
    2: required string Link,
    3: required i32 Begin,
    4: required i32 End,
}

struct ItemMicroAppStruct {
    1: required string AppId,         // Applet ID
    2: required string AppName,       // Applet name
    3: required string Description,   // Applet description
    4: required string Icon,          // Applet Icon
    5: required i16 Orientation,      // Small game screen direction 1 is horizontal screen, 2 is vertical screen
    6: required string Schema,        // Applet schema
    7: required i16 State,            // Status 0 not released 1 released 2 offline
    8: required string Summary,       // brief introduction
    9: required string Title,         // Applet title
    10: required i16 Type,            // Type 1 applet 2 Game 3 article
    11: optional string CardImageUrl, // Video card
    12: optional string CardText,     // Video card tagging copy
    13: optional i32 CardWaitTime,    // How many seconds will the video card be displayed
    14: optional string WebUrl,       // Page URL
}

struct ItemNationalTaskLinkStruct {
    1: required i64 Id,
    2: required string Title,
    3: required string SubTitle,
    4: required UrlStruct AvatarIcon,
    5: optional string WebUrl,           //H5 landing page
    6: optional string OpenUrl,
}

struct LynxButtonStruct {
    1: optional string ButtonBackgroundColor,
    2: optional string Source,
    3: optional string Title,
    4: optional string ImageUrl,
    5: optional string LiveGroupId,
}

struct LynxRawDataStruct {
    1: optional string Refer,                 // Label lynx data sources
    2: optional string WebTitle,              // Landing page title
    3: optional i32 TemplateType,             // Different business categories
    4: optional string OpenUrl,               // Apply direct links
    5: optional LynxButtonStruct ButtonInfo,  // Button data
    6: optional string WebUrl,                // Landing page address
}

struct ItemLiveAppointmentStruct {
    1: optional string WebUrl,                 // Jump landing page of reservation component
    2: optional string Type,                   // type
    3: optional LynxRawDataStruct LynxRawData, // Transparent transmission from client to front end
    4: optional string TemplateUrl,            // Render templates
    5: optional i32 ShowButtonSeconds,         // Display timing of control components
    6: optional i32 Position,
    7: optional i64 ButtonStyle,
}

struct ItemNationalTaskStruct {
    1: optional ItemNationalTaskLinkStruct NationalTaskLinkInfo,    // National task link information
    2: optional ItemLiveAppointmentStruct NationalLiveAppointment,  // National task live booking component information
}

struct ItemOpenPlatformStruct {
    1: required string Id
    2: required string Name
    3: optional UrlStruct Icon
    4: optional string Link
    5: optional string RawData
    6: optional string ClientKey
    7: optional string ShareId
    8: optional bool ShowAnchorName // HAS and SHOW the OpenPlatform anchor name.
}

struct ItemOtherStruct {
    1: optional i32 BodydanceScore,
    2: optional string DescLanguage,
    3: optional string ExtraInfo,
    4: optional string FaceInfo,
    5: optional i64 ForwardCommentId,
    6: optional list<string> GeoFencing,
    7: optional string LandingPageUrl,
    8: optional string MiscRawData,
    9: optional i64 OriginCommentId,
    10: optional i64 OriginItemId,
    11: optional i64 PreForwardItemId,
    12: optional i32 Rate,
    13: optional string RelationLabel,
    14: optional i64 RelationLabelUserId,
    15: optional list<string> SiblingDescs,
    16: optional list<string> SiblingNames,
    17: optional string SortLabel,
    18: optional i32 SourceAppId,
    19: optional i32 SourceType,
    20: optional string StickerIds,
    21: optional i32 TakeDownReason,
    22: optional string TakeDownDesc,
    23: optional list<TextExtraStruct> TextExtraInfos,
    24: optional double TrailerStartTime,
    25: optional enum.VideoDistributeTypeEnum VideoDistributeType,  // Distribution type
    26: optional i32 AdSource,
    27: optional string Ancestor,
    28: optional string TimerInfo,
    29: optional string Deprecated29,
    30: optional string Deprecated30,
    31: optional i32 IsPreview,
    32: optional ItemReviewResultStruct ReviewResultInfo, // Video audit status notification
    33: optional bool ShowShareLink,
    34: optional MaskStruct MaskInfo,                   // report mask info (deprecated), use MaskInfos
    35: optional i64 CommentGID,
    36: optional ItemVideoReplyStruct VideoReplyInfo,
    37: optional string NearbyLabel,
    38: optional i64 LiveId                             // Room ID of submission for live playback, highlight and screen recording
    39: optional list<enum.ImprTagEnum> ImprTags,  // Item identification with special weight reduction requirements
    40: optional string District,                       // County Information of item
    41: optional string IP,                             // IP of item creation
    42: optional list<string> PersonGeoFencing,         // The list of regions where the video is distributed.
    43: optional i32 SpecialMode,                       // The special mode of item, judgment comes from the item extra is_teen_video field
    44: optional ActivityStruct ActivityInfo,           // Dynamic information
    45: optional bool IsFamiliar,                       // is item from familiar
    46: optional string LiveType,                       // Live playback type, live_ replay,live_ highlight,live_ record etc..
    47: optional string PostBillboardType,              // Report type, eg: Star_ challenge,
    48: optional i32 IsStory,
    49: optional bool CoverOverChange,                  // Detection results of dynamic cover amplitude change by TEM
    50: optional bool LightningGuide,                   // Lightning snapshot start guide
    51: optional i32 StoryTTL,                          // story ttl
    52: optional string City,                           // L2 information of item
    53: optional list<MaskStruct> MaskInfos,            // video mask infos
    54: optional VideoCreatorInfoStruct VideoCreatorInfo, // check the video is created by older ios version or not, if true, the client should trim the origin-stitch-part, refer to: https://bytedance.feishu.cn/docs/doccnC0VxswUyUdbmlMBdJYuRVb
    // 55: place holder
    56: optional list<ItemQuestionInfo> QuestionList, // q&a forum info
    57: optional string ContentDesc,                    // caption for content
    58: optional list<TextExtraStruct> ContentDescExtra, // extra info for content caption
    59: optional string CvInfo, // CvInfo from Extra field in Item struct
    60: optional i64 FollowUpPublishFromId, // the original item id which the follow up publish video is shot from, more info https://bytedance.feishu.cn/docs/doccniUL5i3GwexwuzCiFrfwc0e
    61: optional bool DisableSearchTrendingBar, // whether to disable search trending bar on the bottom of the video player, more info https://bytedance.feishu.cn/docs/doccnp0B6yyBeEuPAT4f0d81oEd#
    62: optional ItemGroupIdListStruct GroupIds, // the group id list
    63: optional string BCAdLabelText,           // branded content hashtag new format test
    64: optional string commercial_video_info,   // json string of common commercial info
    65: optional bool AllowReuseOriginalSound // show reuse original sound entrance in music detail page
    66: optional i32 RetryType // // retry video posting type Ex. 0 for normal publish flow, 1 for auto retry. https://bytedance.feishu.cn/docx/doxcnQMNnod8If411uIA4zt1kxh for more info
    67: optional i32 SubRate // safety field
    68: optional map<string, double> CommentInductScores // comment scores of item
    69: optional list<BrandContentAccount> BCAccounts // branded content accounts of item.see more: https://bytedance.feishu.cn/wiki/wikcnvIxVuSJUShrqBnEezB2Lbh
    70: optional PromoteStruct PromoteInfo // promote info
    71: optional bool ShouldAddCreatorTtsWatermarkWhenDownloading // part of creator tts feature - determines if watermark should be present when downloading. https://bytedance.us.feishu.cn/docx/doxusQZWrHS3XYuGXd5omLClL0e for more info
    72: optional bool IsDescriptionTranslatable // Identifies if the description content is translatable.
    73: optional enum.PoiReTagType PoiReTagSignal // Identifies if this item can be re-tag with poi.
    74: optional i64 FollowUpFirstItemId;  // track the original from item id of a follow up published video
    75: optional string FollowUpItemIdGroups;  // track the trace from item id of a follow up published video
    76: optional string ClientText; // the markup_text and text_extra field generated by app client directly
    77: optional string MusicSelectedFrom; // source from music, from ies.item.info
    78: optional i32 MusicTitleStyle; // 0: default disable, but if commercial music, need client check; 1: hide title, ref: https://bytedance.feishu.cn/wiki/wikcneRWWSfKvvVkpk4LlSnGkfe
    79: optional TextToSpeechStruct TextToSpeechInfo; // tts related fields should be wrapped into TextToSpeechStruct
    80: optional VoiceChangeFilterStruct VoiceChangeFilterInfo; // voice change filter related fields should be wrapped into VoiceChangeFilterStruct
    81: optional i32 IsOnThisDay; // if a video is created from past memory entrance, 0:no 1:yes, ref: https://bytedance.feishu.cn/docx/JxNodsNOTok1ZTxKqrjcvHjWncf
    // 82: deprecated
    83: optional ItemCreateAttributeStruct ItemCreateAttribute; // some attributes about creating item
    84: optional bool IsTikTokStory; // decide if a post is tiktok story
    85: optional bool FilterUnfriendlyUserComments; // decide if filter unfriendly user comment https://bytedance.feishu.cn/docx/PtUVdVcPNoch3IxldwGc7A60nrf
    86: optional bool FilterStrangerComments; // decide if filter unfriendly user comment https://bytedance.feishu.cn/docx/PtUVdVcPNoch3IxldwGc7A60nrf
    87: optional bool IsCSEFiltered; // Is Filtered by CSE
    88: optional bool IsStoryToNormal; // decide if a post is switched by story
    89: optional string batch_id; // ID of each batch post
    90: optional i32 batch_index; // post position of the batch post
    91: optional string MainArchCommon; // video arch control related field, tech solution doc: https://bytedance.feishu.cn/docx/QsTAdFQoAo9DptxueuqcJXoMn0f
}

struct ItemCreateAttributeStruct {
    1: optional i32 Original,       // how the item is created 0:upload, 1: shoot
    2: optional i32 IsMultiContent, // Whether the post is composed by multiple footage
    3: optional string ContentType, // The content type of item, can be "photo_canvas"，"multi_photo"，"slideshow"，"now", more specific than aweme_type.
    4: optional string ShootTabName, // The name of tab when you're shooting a video or photo, the value can be "photo"、"story"、"now".
}

struct PromoteStruct {
    1: optional i32 HasPromoteEntry, // item has promote entry or not 1 - show; 2 - gray; 3 - not show
    2: optional string PromoteToastKey // if promote entry is gray, pop a toast. starling key
    3: optional string PromoteToast // if promote entry is gray, pop a toast. starling writing
}

// more info:https://bytedance.feishu.cn/docs/doccnG2hnFu09ZuFE8ZgswfI4ib
struct ItemGroupIdListStruct {
    1: optional list<i64> GroupIdList0, // the group id list 0 for dedup
    2: optional list<i64> GroupIdList1, // the group id list 1 for dedup
}

struct ItemRelationLabelStruct {                                // Main feed focus relation tag
    1: optional i64 CommentId;                                  // Tag related comment ID
    2: optional string Text;                                    // Label suffix text
    3: optional enum.RelationLabelTypeEnum Type;           // Label type
    4: optional list<i64> UserIds;                              // Tag related user ID list
    5: optional string TabText;                                 // Double column display label text
}

struct ItemQuestionInfo {
    1: optional i64 Id,
    2: optional i64 ItemId,
    3: optional i64 UserId,
    4: optional string Username,
    5: optional string Content,
    6: optional UrlStruct UserAvatar,
}

struct ItemPromotionStruct {                         // Used before 580
    1: required i64 Id,                              // Promotion ID
    2: required i32 Clicks,                          // Hits
    3: required i64 CosFee,                          // commission
    4: required double CosRadio,                     // Share ratio
    5: required UrlStruct CoverUrl,                  // Cover map
    6: required string ElasticIntroduction,          // An editor's Guide to talent
    7: required list<UrlStruct> ElasticImageUrls,    // Pictures that can be edited by talent
    8: required string ElasticTitle,                 // Can edit title
    9: required string Extra,
    10: required i64 Gid,                            // Commodity ID
    11: required i64 ItemType,
    12: required bool IsFavorited,                   // Collection or not
    13: required string LandingPageUrl,              // Landing page
    14: required i64 Price,                          // price
    15: required i64 MarketPrice,                    // Market price
    16: required i64 Sales,                          // sales volumes
    17: required string Title,                       // Product title
    18: required i32 Views,                          // Views
    19: optional i64 LastAwemeId,                    // Last bound video ID

    // 220及以后版本新增字段
    20: required i32 ElasticType,                    // General / new / recommended
    21: required string H5Url,                       // Product distribution H5 link
    22: required list<UrlStruct> Images,             // Original multiple pictures of goods
    23: required list<string> Labels,                // Custom label
    24: required i32 Source,                         // Source of goods
    25: required string TitlePrefix,                 // Title Prefix
    26: optional i32 Rank,                           // Ranking of good things
    27: optional string RankUrl,                     // Good thing list jump link
    28: optional PromotionVisitorStruct VisitorInfo, // Visitor information
}

struct ItemReviewResultStruct {
    1: required i32 Status,                         // 0 normal; 1 audit off the shelf; 2 audit self see
    2: optional bool ShouldTell,                    // Can you tell me
    3: optional string DetailUrl,                   // Details page H5
    4: optional string VideoDetailNoticeBottom,     // Red button text under video
    5: optional string VideoDetailNotice,           // Prompt text in the middle of the video
    6: optional string CoverNotice,                 // Personal page under the cover layer text
}

struct ItemRiskStruct {
    1: required i32 Type,
    2: required string Content,
    3: required bool Sink,
    4: required bool Vote,
    5: required bool Warn,
    6: optional bool Notice,
    7: optional string Url,
}

struct ItemSimplePromotionStruct { // after 580
    1: required string RawData,    // Commodity raw data
}

struct ItemSimpleShopSeedingStruct {
    1: required string RawData, // Original data of planting anchor point
}

struct ItemStatisticsStruct {
    1: required i32 CommentCount,
    2: required i32 DiggCount,
    3: required i32 PlayCount,
    4: required i32 ShareCount,
    5: required i32 FakeDiggCount,
    6: required i32 FakePlayCount,
    7: required i32 ForwardCount,
    8: required i32 DownloadCount,
    9: optional enum.CountStatusEnum CountStatus, // count info (except comments)
    10: optional enum.CountStatusEnum CommentCountStatus, // comment count info
    11: optional i32 WhatsAppShareCount,
    12: optional i64 CommentCountI64, // The original I32 type fields do not meet the needs of business growth. Please use the i64 version for relevant count fields
    13: optional i64 DiggCountI64,
    14: optional i64 PlayCountI64,
    15: optional i64 ShareCountI64,
    16: optional i64 FakeDiggCountI64,
    17: optional i64 FakePlayCountI64,
    18: optional i64 ForwardCountI64,
    19: optional i64 DownloadCountI64,
    20: optional i64 WhatsAppShareCountI64,
    21: optional i64 CollectCountI64,
}

struct VideoMuteStruct {
    1: optional bool   IsMute,
    2: optional string MuteDesc,
    3: optional string MuteDetailUrl, // notice tag url
    4: optional string MuteDetailNoticeBottom, // notice tag text
    5: optional bool IsCopyrightViolation,
    6: optional enum.AudioChangeStatusEnum AudioChangeStatus,
}

struct StitchPermissionStruct {
    1: optional i32 Status, // 0: stitchable, 1: disable(gary out), 2:hide
    2: optional string Reason,
}

struct ItemStatusStruct {
    1: required bool AllowComment,
    2: required bool AllowShare,
    3: required i32 DownloadStatus,    // 0:allow_download, 1:ad_prevent
    4: required bool InReviewing,
    5: required bool IsDelete,
    6: required bool IsHotReviewed,
    7: required bool IsPrivate,
    8: required bool IsReviewed,
    9: required bool IsUserDigged,
    10: required bool IsVideoProhibited,
    11: required i32 PrivateStatus,    // 0: public, 1: private, 2: visible to friends
    12: required bool SelfSee,
    13: required i32 ValidFansStatus,
    14: required bool WithFusionGoods,
    15: required bool WithGoods,       // with a product or not
    16: required bool WithPromotionMusic,
    17: required i32 SecondReviewStatus,
    18: optional i32 MusicEditStatus, //music edit status
    19: optional VideoMuteStruct VideoMute, // video mute
    20: optional bool AllowBeLocated,
    21: optional i32 ThirdReviewStatus,
    22: optional StitchPermissionStruct StitchPermission, // use for stitch permission control
    23: optional enum.InsightStatusEnum VideoInsightStatus, // refer enum, use to enable/disable video insight H5 page when creator views from profile page, see: https://bytedance.us.feishu.cn/docx/ILdodzikloeb0bxYZyduVYngsRo
}

struct ItemStarAltasLinkStruct {
    1: required i64 Id,
    2: required i64 OrderId,
    3: required string Title,
    4: required string SubTitle,
    5: required UrlStruct AvatarIcon,
    6: optional string WebUrl,      //H5 landing page
    7: optional string OpenUrl,
}

struct ItemTalentStruct {
    1: optional string GoodsRecUrl,                         // New version of talent add product link
    2: optional string ManageGoodsUrl,                      // Add product link
    3: optional i64 StarAtlasOrderId,                       // Star map order ID
    4: optional i32 StarAtlasStatus,                        // Star map status
    5: optional ItemStarAltasLinkStruct StarAtlasLinkInfo,  // Star map link information
    6: optional i32 TcmStatus,                              // TCM order status: 1 - closed
    7: optional bool PreventPrivacy,                        // Disable visibility operations
    8: optional string PreventPrivacyReason,                // Why visibility operations are prohibited
    9: optional enum.TCMReviewStatusEnum TCMReviewStatus, // TCM review status
}

struct ItemVideoReplyStruct{
    1: optional i64 AwemeId, // id of the video being replied to
    2: optional i64 CommentId, // id of the ecomment being replied to
    3: optional i64 AliasCommentId, // comment id corresponding to the reply video
    4: optional string UserName, // username of the user who created the original comment
    5: optional string CommentMsg, // comment message corresponding to comment_id
    6: optional i64 CommentUserId, // id of the user who created the original comment
    7: optional UrlStruct UserAvatar, // avatar thumbnail of the user who created the original comment
    8: optional bool IsFavorited, // not in use
    9: optional bool CollectStat, // whether or not comment is collected

}

struct ItemDouplusToastStruct {
    1: required i64 Id,              // toast Id
    2: required i64 ContentId,       // content Id
    3: required i32 Type,            // toast type
    4: required string Content,      // toast content
    5: optional string RedirectUrl,  // url
}

struct ItemDouplusStruct {
    1: optional ItemDouplusToastStruct ToastInfo,  // toast info
}

struct ItemShareTaskStruct {
    1: required i32 ShowTimestamp,      // Task presentation time, unit: ms
    2: required i32 CoinCount,          // Number of gold coins for task reward
}

// Music Structs
struct MusicAudioStruct {
    1: required i32 Duration,
    2: required i32 EndTime,
    3: required i32 StartTIme,
    4: required UrlStruct PlayUrl,
    5: optional UrlStruct AudioTrackUrl,
    6: optional UrlStruct CoverHdUrl,
    7: optional UrlStruct CoverMediumUrl,
    8: optional UrlStruct CoverLargeUrl,
    9: optional UrlStruct CoverThumbUrl,
    10: optional UrlStruct EffectsDataUrl,
    11: optional enum.LyricTypeEnum LyricType,
    12: optional string LyricUrl,
    13: optional double PreviewStartTime,
    14: optional double PreviewEndTime,
    15: optional i32 AuditionDuration, //Audition duration
    16: optional i32 ShootDuration, //Record duration
    17: optional i32 VideoDuration, // duration can be used in video
    18: optional DurationHighPrecisionStruct DurationHighPrecision // high precision structs used for song duration
    19: optional i64 PlayUrlExpiredAt, // unix time, only for ugc music
    20: optional FullSongStruct FullSong, // fields for full song
}

struct MusicExternalStruct {
    1: required string H5Url,
    2: required string PartnerName,
    3: required string PartnerSongID,
    4: required string SongKey,
    5: optional string ExternalAppLink,
    6: optional string ExternalDeepLink,
    7: optional string LabelId,
}

struct MusicMuteStruct {
    1: optional bool      IsVideoMute,
    2: optional list<enum.MusicUnshelveReasonEnum> MuteReasonTypes,
}

struct DspAuthToken {
    1:optional AppleMusicToken AppleMusicToken,
}

struct AppleMusicToken {
    1: optional string DeveloperToken,
    2: optional string UserToken,
}

struct TT2DspInfoStruct {
    1: optional enum.DspPlatform Platform,
    2: optional string SongId,
    3: optional bool PlatformSelectByUser,
    4: optional DspAuthToken Token,
    5: optional i64 ShowStrategy,
}

struct MusicOtherStruct {
    1: optional UrlStruct BodydanceChallengeId,
    2: optional i64 BindedChallengeId,
    3: optional UrlStruct BodydanceUrl,
    4: optional string Extra,
    5: optional UrlStruct StrongBeatUrl,
    6: optional list<string> UnshelveCountries,
    7: optional string InternalExtra,
    8: optional list<i64> UGCMappingPGCClipIDs,
    9: optional bool IsMatchedMetadata, // is music metadata from matched PGC sound clip?
    10: optional string ExternalSongSubtitle,
    11: optional bool CanReplaceMusic,
    12: optional MusicMuteStruct MusicMute,
    13: optional UGCMappingSongStruct UGCMappingSong,
    14: optional i64 ExtractedItemID,
    15: optional bool MusicSelectIsSimMusic, // is sim music
    16: optional bool IsShootingAllow,
    17: optional i32 CommercialSceneStatus // status of commercial scene; get from music.GetDetails().GetCommercialSceneStatus()
	18: optional bool IsMajorThreeMusic, // is major 3 music, check by labelID, more info https://bytedance.feishu.cn/docx/ETu6dJWRDoKerjxL9LjchEGGnRf
	19: optional list<TT2DspInfoStruct> TTToDSPInfos,
	20: optional list<DetailStatusInfo> DetailStatusInfoList //  details Status Info List
	21: optional bool HasCommerceRight
	22: optional content_standard_biz.TT2DSPAlbumStruct TT2DSPAlbumInfo,
}

struct DetailStatusInfo {
	1: optional enum.DetailStatusReasonTypeEnum ReasonType, // scene
    2: optional enum.ShortVideoDetailStatusEnum ShortVideoSceneStatus, // Song status in short video scene
}


struct MusicOwnerStruct {
    1: required i64 Id,
    2: required string Nickname,
    3: required string SecretId,
    4: required string Sign,
    5: optional bool VCDNotAuth,
}

struct MusicArtistStruct {
    31: optional i64 UserId,
    32: optional string Nickname,
    33: optional string SecretId,
    34: optional string UniqueID,
    35: optional UrlStruct Avatar,
    36: optional bool IsVerified,
    37: optional enum.MusicArtistTypeEnum EnterType, // Types of artists
    38: optional enum.FollowStatusEnum FollowStatus,
    39: optional bool IsVisible, // Visibility of artists
    40: optional enum.FollowStatusEnum FollowerStatus,
    41: optional bool IsPrivateAccount,
    42: optional bool IsBlock,
    43: optional bool IsBlocked,
    44: optional i16 Status,
}

struct MusicUncertArtistStruct {
    1: required string Name,
}

// NoticeStructs
struct OfficialAssistStruct {
    1: optional string Keyword,
    2: optional string DisplayKeyword,
}

struct NoticeCommentStruct {
    1: optional string ReplyUserID,
    2: optional string ReplyUserNickname,
    3: optional string AliasItemId, // If the reply is a video comment, use this field to Identity the comment video ID
}

struct NoticeDiggStruct {
    1: optional i32 DiggType,
    2: optional bool hasDiggList,
    3: optional i32 RefType,
    4: optional string ForwardID,
    5: optional NudgeInfoStruct NudgeInfo,
}

struct NudgeInfoStruct{
    1: optional i32 NudgeType,
    2: optional string AwemeID,
    3: optional string AuthorID,
    4: optional string CoverUrl,
    5: optional i32 AwemeType,
    6: optional bool IsStory,
    7: optional string NudgeEnterMethod,
}

struct NoticeGameStruct {
    1: optional string Id,
    2: optional string Name,
    3: optional string SubTitle,
    4: optional string Icon,
}

struct NoticeMicroAppStruct {
    1: optional string Id,
    2: optional string Name,
    3: optional string SubTitle,
    4: optional string Icon,
}

struct NoticeCreatorMissionStruct {
    1: optional string Id,
    2: optional string Name,
}

struct NoticeMigrateStruct {
    1: optional string ExportUserID,
    2: optional string ImportUserID,
    3: optional i64 RejectStatus,
    4: optional list<NoticeMigrateTextStruct>  MigrateText,
}

struct NoticeMigrateTextStruct {
    1: optional string text,
    2: optional i64  key,
    3: optional i64  action,
    4: optional string link,
}

struct NoticeShopStruct {
    1: optional string SessionId,
    2: optional string SessionAvatar,
}

struct NoticeTcmStruct {
    1: optional i32 SourceType,
    2: optional string SourceName,
    3: optional UrlStruct SourceIcon,
    4: optional UrlStruct DarkSourceIcon,
}

struct NoticeVCDStruct {
    1: required i32 JumpStatus,
    2: required string SchemaURL,
    3: required string SchemaText,
}

struct NoticeVoteStruct {
    1: optional string OptionText,
    2: optional i32 RefType,
}

struct NoticeQaStruct {
    1: optional i32 qa_type,              // refer to tiktok/idl/api/v1/enum.proto QaNoticeType
    2: optional i64 question_id,
    3: optional i32 question_group_type,  // 0: question to item, 1: question to user
    4: optional i64 answer_id,
    5: optional i64 answer_item_id,
}

// additional info for system notice template with title and content
struct SystemNoticeTemplateStruct {
    1: optional UrlStruct avatar_image_url,      // Not Set for normal assistant in extra
    2: optional UrlStruct avatar_dark_image_url, // Not Set for normal assistant in extra
    3: optional string avatar_schema_url,        // Not Set for normal assistant in extra
    4: optional i32 middle_type,                 // TitleContent(default) / NameContent
    5: optional string content_schema_url,       // root schema will use "content_schema_url" by default
    6: optional i32 right_type,         // empty / arrow / button / image
    7: optional i32 button_type,        // if right is button
    8: optional string button_text,     // if right is button
    9: optional UrlStruct image_url,    // if right is image
    10: optional string schema_url,     // right schema using "schema_url", use "content_schema_url" if empty
    11: optional string business_extra, // business event tracker and extra info, for specific biz need, should not be too large
    12: optional string account_type,   // for event tracker report, Not Set for normal assistant in extra
    13: optional list<TextLinkConfig> content_text_link_configs, // for the text link feature in the content
    14: optional list<NoticeButtonStruct> buttons, // buttons on the buttom
    15: optional string quote_content, // quote content if exist
    16: optional list<string> middle_image_uris, // middle image uris
    17: optional UrlStruct banner_image_url, // banner image uri. for the system notice in system notice box only
    18: optional string banner_schema_url, // the schema url for the banner image in system notice box
}

// template struct for interactive notice
struct InteractiveNoticeTemplateStruct {
    1: optional string root_schema,
    2: optional i32 middle_type,        // NameContentEvent / NameContentEventWithQuote / NameContentEventWithImage
    3: optional string extra_schema,    // Normally for merge types when len(fromUsers)>=2
    4: optional string content,         // notice content, normally translate by starling
    5: optional bool is_event,          // convert style content to style event
    6: optional string quote_content,   // if need quote
    7: optional UrlStruct quote_img,    // if need quote img
    8: optional i32 right_type,         // empty / arrow / button / image
    9: optional i32 button_type,        // if right is button
    10: optional string button_text,    // if right is button
    11: optional UrlStruct image_url,   // if right is image
    12: optional string schema_url,     // right schema using "schema_url"
    13: optional string business_extra, // business event tracker and extra info, for specific biz need, should not be too large
    14: optional string account_type,   // for event tracker report
    15: optional string related_item_id,// for event tracker report
    16: optional string middle_schema_url, // Middle schema uses "middle_schema_url". It's only used for middle quote now.
    17: optional list<TextLinkConfig> content_text_link_configs, // for the text link feature in the content
    18: optional string title,              // middle title
    19: optional UrlStruct avatar_image_url,      // customized left avatar
    20: optional UrlStruct avatar_image_url_dark, // customized left avatar, dark mode
    21: optional string middle_label, // middle label json string, details defined in pb idl
    22: optional string title_extra_action, // middle title action, details defined in pb idl
    23: optional CommentStruct comment_info // for inbox quick reply/digg
    24: optional UrlStruct right_icon_url, // right icon url
    25: optional list<UrlStruct> middle_image_urls, // middle image packed urls
    26: optional list<string> middle_image_schema_urls, // middle image schema urls
    27: optional UrlStruct avatar_badge_url, // avatar badge url
    28: optional UrlStruct avatar_cover_url, // avatar cover url
    29: optional i32 legacy_sub_type, // for non template migration only
    30: optional string toast_text, // toast text
    31: optional string template_content, // template content. e.g. "commented: %v"
    32: optional list<CommentImageStruct> comment_image, // comment photos
    33: optional NudgeInfoStruct NudgeInfo,
    34: optional list<NoticeButtonStruct> middle_bottom_buttons,
    35: optional UrlStruct image_url_dark,
    36: optional UrlStruct dark_mode_quote_image,    // if need dark_mode_quote_image
}

struct TextLinkConfig { // for notice UI template
    1: optional string text,
    2: optional string schema_url,
    3: optional enum.TextLinkTypeEnum type,  // TextLink UI type
    4: optional enum.TextLinkSchemaTypeEnum schema_type,
    5: optional bool need_track,
}

// User Structs
struct UserAvatarDecorationStruct {
    1: required i64 Id,
    2: required string Name,
    3: required UrlStruct SourceUrl,
}

struct UserCertificationStruct {
    1: required i32 CertType,
    2: required i32 OrganizationType,
    3: optional enum.MTCertTypeEnum MTCertType, // Certification type : individual, institution, enterprise
}

struct UserCommerceAccountStruct {
    1: required i32 Level,
    2: optional list<i64> TopItemIds,
    3: optional enum.ProAccountStatus ProAccountStatus, // Pro account status
    4: optional bool BusinessAccount,
}

struct UserCommunityStruct {
    1: required i32 DisciplineStatus,
}

struct UserEcommerceAccountStruct {
    1: required list<i64> TopItemIds,
}

struct UserExtraStruct {
    1: optional string GeneralExtra,
    2: optional string Extra,
}

struct UserIconStruct {
    1: required string AvatarUri,
    2: required UrlStruct AvatarThumbUrl,
    3: required UrlStruct AvatarMediumUrl,
    4: required UrlStruct AvatarLargeUrl,
    5: required UrlStruct AvatarUrl168x168,
    6: required UrlStruct AvatarUrl300x300,
    7: required list<UrlStruct> BackgroundImages,
    8: required UrlStruct VideoIconUrl,
    9: optional UserAvatarDecorationStruct AvatarDecorationInfo,
    10: optional string VideoIconVirtualURI,
    11: optional string SocialAvatarWithoutBackgroundUri,
    12: optional AvatarSizeUrlStruct SocialAvatarWithoutBackground,
    13: optional string DynamicSocialAvatarUri,
    14: optional AvatarSizeUrlStruct DynamicSocialAvatar,
    15: optional list<AvatarCategoryMetaInfo> AvatarMetaInfoList,
}

struct UserOriginalMusicianStruct {
    1: required i32 Count,
    2: required i32 Digg,
    3: required i32 Used,
    4: optional UrlStruct CoverUrl,
    5: optional UrlStruct QrcodeUrl,
}

struct UserOtherStruct {
    1: required bool Blocked,
    2: required enum.FollowStatusEnum FollowStatus,
    3: required i64 RoomId,
    4: optional string RemarkName,
    5: optional list<i64> LabelTypes,
    6: optional i32 FollowerStatus,
    7: optional string RecommendReason,
    8: optional string RoomData,
    9: optional i64 AgeInterval,
    10: optional i64 UgcOffer,
    11: optional i32 ReviewTestTag,
    12: optional i16 RealStatus,
    13: optional list<i64> BanUserFunctions,
    14: optional string UseWxName,
    15: optional string UseWxAvatar,
    16: optional string WxReplaceAvatarUri,
    17: optional string WxReplaceNickname,
    18: optional string Extra,
    19: optional bool IsFamiliar,
    20: optional double FamiliarScore,
    21: optional i32 StoryTag,
    22: optional enum.BlockStatusEnum BlockStatus,
    23: optional map<i64, bool> ValidMiddleUserIDs,  // In some scene user may have related other users, this field decide that whether this related user can be shown.
    24: optional MatchedFriendStruct MatchedFriend,
    25: optional i32 PhoenixStatus,
    26: optional i32 RestrictMode, // 0: Not in RM, 1: turn on by user self, 2: turn on by parent
    27: optional i32 HasLoggedInOnTT,  // 0: default, 1: not logged in, 2: logged in
    28: optional i32 HasLoggedInOnTTM,   // 0: default, 1: not logged in, 2: logged in
    29: optional i32 PublishActiveness,   // user activeness level from FeatureTag
    30: optional list<enum.SpecialAccountType> SpecialAccountList,
    31: optional enum.FriendsStatusEnum FriendsStatus,
    32: optional SpecialAccountStruct SpecialAccount,
    33: optional bool IsInSameNoteExpGroup,
    34: optional list<enum.FollowStatusEnum> CanMessageFollowStatusList,
    35: optional FakeDataStruct FakeDataInfo,
    36: optional UserSparkInfo UserSparkInfo,
}

struct UserSparkInfo {
    1: optional bool IsUnavailableRegionUser,
}

struct FakeDataStruct {
    1: optional string FakeNickname,
    2: optional string FakeUsername,
    3: optional bool IsFakeAvatar,
    4: optional bool IsFakeBio,
}

struct SpecialAccountStruct {
    1: optional TTNowSpecialAccountStruct TTNow,
}

struct TTNowSpecialAccountStruct {
     1: optional enum.AccountLogStatusEnum TTNowLogStatus,
}

struct MatchedFriendStruct {
   1: optional string RecType,
   2: optional string RecommendReason;
   3: optional MutualStruct MutualStruct;
   4: optional string RelationType;
   5: optional string SocialInfo;
   6: optional ExternalRecommendReasonStruct ExternalRecommendReason; // Copy Optimisation. Display third-party username.https://bytedance.feishu.cn/docs/doccnxNI41p1QclSWDFb0bZthZd
}

struct ExternalRecommendReasonStruct{
    1: optional string Reason;
    2: optional string HashedPhoneNumber;
    3: optional string ExternalUsername;
}

struct MutualStruct {
    1: optional enum.MutualType MutualType;
    2: optional list<MutualUserStruct> UserList;
    3: optional i32 Total;
}

struct MutualUserStruct {
    1: optional string SecUid;
    2: optional string Nickname;
    3: optional UrlStruct AvatarMedium;
    4: optional i64 UserId;
}

struct UserStatisticsStruct {
    1: required i64 AwemeCount,
    2: required i64 FavoritingCount,
    3: required i64 FollowerCount,
    4: required i64 FollowingCount,
    5: required i64 StoryCount,
    6: required i64 TotalFavorited,
    7: optional i64 XmasUnlockCount,
    8: optional enum.CountStatusEnum FollowCountStatus,
    9: optional i64 HistoryMaxFollowerCount,
}

struct UserSocialStruct {
    1: required i32 AppleAccount,
    2: required i32 FacebookExpireTime,
    3: required string GoogleAccount,
    4: required bool HasFacebookToken,
    5: required bool HasTwitterToken,
    6: required bool HasYoutubeToken,
    7: required bool IsBindedWeibo,
    8: required string InsId,
    9: required i32 TwitterExpireTime,
    10: required string TwitterId,
    11: required string TwitterName,
    12: required string WeiboName,
    13: required string WeiboSchema,
    14: required string WeiboUrl,
    15: required string WeiboVerify,
    16: required string YoutubeChannelId,
    17: required string YoutubeChannelTitle,
    18: required i32 YoutubeExpireTime,
}

struct UserSettingStruct {
    1: required bool AcceptPrivatePolicy,
    2: required string AccountRegion,
    3: required i32 CommentFilterStatus
    4: required i32 CommentSetting,
    5: required i32 Creatorlevel,
    6: required string CvLevel,
    7: required i64 DownloadPromptTimestamp,
    8: required i32 DownloadSetting,
    9: required i32 DuetSetting,
    10: required list<string> Geofencing,
    11: required bool HasInsights,
    12: required bool HasRegisterNotice,
    13: required bool HideLocation,
    14: required bool HideSearch,
    15: required i32 LiveAgreement,
    16: required i64 LiveAgreementTime,
    17: required i32 LiveRecLevel,
    18: required string Location,
    19: required i32 Mode,
    20: required bool NicknameLock,
    21: required bool NeedRecommend,
    22: required i32 Period,
    23: required map<string, i32> PolicyVersion,
    24: required bool PreventDownload,
    25: required i32 Rate,
    26: required i32 ReactSetting,
    27: required i32 RealnameVerifyStatus,
    28: required i32 RiskFlag,
    29: required i64 ReflowPageGid,
    30: required i64 ReflowPageUid,
    31: required string Deprecated31,
    32: required string Deprecated32,
    33: required i32 Deprecated33,
    34: required bool ShieldCommentNotice,
    35: required bool ShieldDiggNotice,
    36: required bool ShieldFollowNotice,
    37: required bool ShieldNeiguang,
    38: required bool SpecialLock,
    39: required i32 StoryReplyPermissionStatus
    40: required bool Deprecated40,
    41: required bool UpdateBefore,
    42: required i32 IsolationLevel,
    43: optional enum.FavoriteListPermissionEnum FavoriteListPermission,
    44: optional list<string> CommentFilterWords,
    45: optional list<string> GeoFilter,
    46: optional enum.FavoriteListPermissionEnum FavoriteOnItemPermission,
    47: optional bool AllowBeLocated,
    48: optional i32 StitchSetting, // Stick setting of user dimension
    49: optional i32 VideoGiftSetting,
    50: optional i32 PhotosensitiveVideosSetting,
    51: optional i32 SugToContacts,
    52: optional i32 SugToFBFriends,
    53: optional i32 SugToMutualConnections,
    54: optional i32 SugToWhoShareLink,
    55: optional i32 SugToInterestedUsers,
    56: optional string SelectedTranslationLanguage,
    57: optional list<i32> TnsAccountLabels,  // The identification of some information displayed by tns
    58: optional i32 HidePostSetting,
    59: optional bool ShowPlaylist, // privilege, to show playlist, add to settings in order to load in fyp
    60: optional i64 DSPPreference, // TT to DSP user platfor preference
    61: optional list<string> CommentGeoFilter,
    62: optional list<string> MessageGeoFilter,
    63: optional bool AuthorEnableFilterAllComments,
}

struct Deprecated3Struct {
    1: required bool Deprecated1,
    2: required bool Deprecated2,
    3: required bool Deprecated3,
    4: required bool Deprecated4,
    5: optional i32  Deprecated5,
    6: optional bool Deprecated6,
    7: optional i64  Deprecated7,
    8: optional i64  Deprecated8,
}

struct UserVerificationStruct {
    1: required i64 AuthorityStatus,
    2: required string CustomVerify,
    3: required bool EnterpriseVerify,
    4: required string EnterpriseVerifyReason,
    5: required bool IsVerified,
    6: required i32 LiveVerify,
    7: required i32 VerificationType,
    8: required string VerifyInfo,
    9: optional bool HasCert,
}

struct UserUnreadVideoStruct {
    1: optional i32 Count,
    2: optional i64 LatestTime,
    3: optional list<i64> ItemIds,
}

// Meta Structs
struct ChallengeStruct {
    1: required i64 Id,
    2: required i64 Type,
    3: required string ChaName,
    4: required i64 CreateTime,
    5: required string Desc,
    6: required string Schema,
    7: required i64 UserId,
    8: required i32 Status,
    9: optional ChallengeMaterial MaterialInfo,
}

struct ItemStruct {
    1: required i64 Id,
    2: required i32 AppId,
    3: required i64 ChallengeId,
    4: required i64 CreateTime,
    5: required string Desc,
    6: required i64 GroupId,
    7: required enum.MediaTypeEnum MediaType,
    8: required i64 MusicId,
    9: required string Region,
    10: required i32 Status,
    11: required i32 Type,
    12: required i64 UserId,
    13: required i32 VrType,
}

struct MusicStruct {
    1: required i64 Id,
    2: required string Album,
    3: required string Author,
    4: required i64 CreateTime,
    5: required string OfflineDesc,
    6: required i16 LabelId,
    7: required string SchemaUrl,
    8: required i16 SourcePlatform,
    9: required i16 Status,
    10: required string Title,
    11: required i64 UsedCount,
    12: optional enum.MusicUnusableReasonTypeEnum UnusableReasonType,
    13: optional bool IsAudioURLWithCookie,
    14: optional list<BitrateStruct> MultiBitRatePlayStruct,
    15: optional i64 ShortVideoDetailStatus,
    16: optional i16 RecommendStatus,
    17: optional i64 VideoStatus,
    18: optional string VideoVolumeInfo,
}

struct NoticeStruct {
    1: required i64 ID,
    2: required i32 Type,
    3: required i64 CreateTime,
    4: optional string Title,
    5: optional string Content,
    6: optional string SchemaUrl,
    7: optional UrlStruct ImageUrl,
    8: optional i32 SubType,
    9: optional i32 ItemMediaType,
    10: optional i32 MergeCount,
    11: optional bool Pending,
    12: optional list<i64> FromUserIDs,
    13: optional i64 TaskID,
    14: optional i64 RefID,
    15: optional i64 ObjectId,
    16: optional string NoticeLabelText,
    17: optional string WebUrl,
    18: optional string LogExtra,
    19: optional i32 NoticeLabelType,
    20: optional i32 ActionApp,
    21: optional bool HasRead,
    22: optional string ContentSchemaUrl,
    23: optional string SchemaText,
    24: optional bool FromNew, // Is the message from the new system
    25: optional string MessageExtra,  // The embedded point fields defined by the business side are in JSON format
    26: optional string TemplateId,
    27: optional i32 UnsubscribeLabel,
    28: optional list<FrequencyConfig> FrequencyConfigs,
    29: optional i32 RelevantScore,
    30: optional bool IsFriend, // Is the message from friend
    // dark mode item cover
    31: optional UrlStruct ImageUrlDark,

    128: optional OfficialAssistStruct OfficialAssistInfo,
    129: optional NoticeDiggStruct DiggInfo,
    130: optional NoticeVoteStruct VoteInfo,
    131: optional NoticeCommentStruct CommentInfo,
    132: optional NoticeShopStruct ShopInfo,
    133: optional NoticeGameStruct GameInfo,
    134: optional NoticeMigrateStruct MigrateInfo,
    135: optional XSpaceStruct XSInfo,
    136: optional NoticeTcmStruct TcmInfo,
    137: optional NoticeCreatorMissionStruct CreatorMissionInfo,
    138: optional ShopServiceStruct ShopServiceInfo,
    139: optional NoticeMicroAppStruct MicroAppInfo,
    140: optional NoticeQaStruct QaInfo,
    141: optional SystemNoticeTemplateStruct Template,
    142: optional InteractiveNoticeTemplateStruct InteractiveTemplate,
}

struct UserStruct {
    1: required i64 Id,
    2: required i32 AppId,
    3: required string BindPhone,
    4: required string Birthday,
    5: required enum.ConstellationTypeEnum Constellation,
    6: required i64 CreateTime,
    7: required string Email,
    8: required enum.Deprecated1Enum Gender,
    9: required string Language,
    10: required string Nickname,
    11: required string Phone,
    12: required string Region,
    13: required string SecretId,
    14: required i64 ShortId,
    15: required string Signature,
    16: required i16 Status,
    17: required string UniqueId,
    18: required i64 UniqueIdModifyTime,
    19: optional string RegistrationCountry,
    20: optional list<SignatureExtra> SignatureExtraList,
}

struct SignatureExtra {
    1: required i32 Start,
    2: required i32 End,
    3: required enum.TextTypeEnum Type,
    4: optional string UserId,
    5: optional string SecUid,
}

struct MixStatisStruct {
    1: optional i64 PlayVV,
    2: optional i64 CollectVV,
    3: optional i64 CurrentEpisode,
    4: optional i64 UpdatedToEpisode,
    5: optional i64 HasUpdatedEpisode,
}

struct MixStatusStruct {
    1: required i32 Status,
    2: optional i32 IsCollected,
}

struct MixStruct {
    1: required i64 Id,
    2: required string Name,
    3: optional UrlStruct CoverUrl,
    4: optional UrlStruct IconUrl,
    5: optional MixStatusStruct Status,
    6: optional MixStatisStruct Statis,
    7: optional string Desc,
    8: optional UserStruct Author, //Deprecated
    9: optional string Extra,
    10: optional i64 UserId,
    11: optional i32 Type,
	12: optional i64 CreateTime,
	13: optional i64 UpdateTime,
}

struct MixOtherStruct {
    1: optional map<i64, i32> EpisodeMap,
    2: optional AutoMixAuthorStruct AutoMixAuthor,
}

struct AutoMixAuthorStruct {
    1: required string title;
    2: required string schema_url;
}

struct ShopServiceStruct {
    1: optional i32 BizId,
    2: optional i32 BizType,
}

struct UGCMappingSongStruct {
    1: required i64 SongID,
    2: required string Title,
    3: required string Author,
    4: optional list<i64> ClipIDs,
    5: optional string MixedTitle,
    6: optional string MixedAuthor,
    7: optional bool IsNewReleaseSong,
    8: optional i64 GroupReleaseTime,
    9: optional UrlStruct MediumCoverUrl,
}

struct MusicPerformerStruct {
    1: required i64 Id,
    2: optional string Name,
    3: optional MusicAvatarStruct Avatar,
}

struct MusicChorusStruct {
    1: required i32 StartMS,
    2: required i32 DurationMS,
}

struct MusicAvatarStruct {
    1: optional UrlStruct AvatarThumb,     // 100*100
    2: optional UrlStruct AvatarThumbnail,   // 168*168
    3: optional UrlStruct AvatarMedium,   // 300*300
    4: optional UrlStruct AvatarLarge,    // 720*720
    5: optional UrlStruct AvatarHd,      // 1080*1080
}

struct MappingSongExtra {
    1: optional i64 TTPreSaveStartTime,
    2: optional i64 TTNewReleaseStartTime,
}

struct MappingSongStruct {
    1: required i64 SongID,
    2: optional string Title,
    3: optional string Author,
    4: optional i32 FullDuration,
    5: optional UrlStruct MediumCoverUrl,
    6: optional string h5_url,
    7: optional list<MusicExternalStruct> ExternalInfos,
    8: optional list<MusicPerformerStruct> PerformerInfos,
    9: optional MusicChorusStruct ChorusInfo,
    10: optional MappingSongExtra Extra,
    11: optional i64 GroupReleaseTime, //New Release feature PRD:https://bytedance.feishu.cn/docx/JaJFdE8m3oJ7l7xI3mxcTMiCndy
    12: optional bool IsNewReleaseSong, //New Release feature PRD:https://bytedance.feishu.cn/docx/JaJFdE8m3oJ7l7xI3mxcTMiCndy
    13: optional list<MusicArtistStruct> ArtistInfos, // by artist phase 2 PRD: https://bytedance.sg.feishu.cn/docx/XDcfd5FSSoPsY7x2OjwlS5wlgeh
    14: optional list<MusicUncertArtistStruct> UncertArtistInfos, // MDP Artist Display PRD:https://bytedance.sg.larkoffice.com/docx/N6Hdd6h3ZoR93SxgYswlJWBigtg
}

struct MusicDetailStatusInfo {
    1: optional enum.InteractiveMusicStreamingDetailStatusEnum InteractiveMusicStreamingSceneStatus, // Song status in interactive music streaming scene
}

struct UnFilteredMappingSongStruct {
    1: required i64 SongID,
    2: optional string Title,
    3: optional string Author,
    4: optional i32 FullDuration,
    5: optional list<MusicExternalStruct> ExternalInfos,
    6: optional MappingSongExtra Extra,
    7: optional MusicDetailStatusInfo Details,
    8: optional list<i64> ArtistIDs,
}

struct StoryItemStruct{
    1: required i64 StoryId,
    2: required bool Viewed,
    3: required i64 ExpiredAt,
    4: optional i64 TotalComments,
    5: optional bool IsOfficial,
    6: optional i64 ViewerCount,
    7: optional bool ChatDisabled,
    8: optional enum.StoryType StoryType,
	9: optional i64 StoryStyleVersion,
}

struct UserStoryStruct{
    1:required i32 Total,
    3:required i32 Postion,
    4:required list<StoryItemStruct> StoryItemList,
    5:required i64 MinCursor,
    6:required i64 MaxCursor,
    7:required bool HasMoreAfter,
    8:required bool HasMoreBefore,
    9:required bool AllViewed,
    10:optional i64 LastStoryCreatedTime, // milliseconds
    11:optional bool IsPostStyle, // for post style single story
    12:optional list<StoryLiteMetadata> AllStoryLiteMetadata, // user all story metadata
}

struct StoryLiteMetadata {
    1:optional i64 ItemID,
    2:optional i64 ProgressBarNum,
    3:optional bool Viewed,
    4:optional i32 AwemeType,
    5:optional i64 ExpireAt,
}

struct ItemBottomBarStruct {
    1: optional string Content,
    2: optional enum.BottomBarTypeEnum Type,
    3: optional UrlStruct Icon,
}

struct ItemBoostStruct {
    1: optional enum.BoostTypeEnum Type,
    2: optional string Label,
    3: optional string Color, // background color
    4: optional string ColorText,
}

struct BubbleStruct {
    1: required i32 Biz,           // business source
    2: required i32 Type,          // style type
    3: optional string Text,       // text displayed to user
    4: optional string SchemaUrl,  // target page url for click
    100: optional string LogExtra, // for app event metrics
}

struct CreatorPlaylistStruct {
    1: required i64 MixId, // playlist id
    2: optional i64 UserId, // creator
    3: optional string Name, // name of playlist
    4: optional i32 Status, // playlist status
    5: optional i32 Review, // review status
    6: optional string NameInReview, // mix name in moderation review, can only be seen by the owner
    7: optional bool IsDefaultName, // whether the current playlist is using default name
    8: optional i64 Index, // index of the video in this mix
    9: optional i64 ItemTotal, // total number of item
}

struct Stitch {
    1: optional i64 OriginalAwemeId, // the original video stitched from
    2: optional i64 TrimStartTime,  // The time when video trim starts
    3: optional i64 TrimEndTime,  // The time when video trim ends
}

struct DuetInfo {
    1: optional i64 OriginalItemId, // The original video item ID duetted from
    2: optional i64 DuetLayout, // The duet layout, e.g. new_left
}

struct GreenScreenMaterial {
    1: optional enum.GreenScreenType Type,
    2: optional i64 StartTime,
    3: optional i64 EndTime,
    4: optional string ResourceId,
    5: optional string AuthorName,
    6: optional string EffectId,
}

struct DeprecatedGreenScreenMaterialStruct {
    1: optional enum.GreenScreenType Type,
    2: optional i64 StartTime,
    3: optional i64 EndTime,
    4: optional string ResourceId,
    5: optional string AuthorName,
    6: optional string EffectId,
}

//  Context of the creator such as device_platform, os_version, app_version, etc...
struct VideoCreatorInfoStruct {
    1: optional string DevicePlatform,
    2: optional i32 ClientVersion,
}

struct TrendingBarStruct {
    1: required UrlStruct IconUrl, //left icon url
    2: required string Display, //display text
    3: required UrlStruct ButtonIconUrl, //right button icon url
    4: required string Schema, //link
    5: optional i64 EventKeywordId,//event id
    6: optional string EventKeyword, //event name
    7: optional string EventTrackingParam, //client log param
}

struct OriginalLanguageStruct {
    1: required string Lang, // the language name from video arch subtitle management system
    2: required i64 LanguageId, // map<lang, language_id> is maintained by vcloud. https://bytedance.feishu.cn/docs/doccnL3XBAihnFwawShre7pjgyh
    3: optional string LanguageCode, // the language code from lab which aligned with translation language
    4: optional bool CanTranslateRealtime, // whether the original caption supports realtime translation
    5: optional enum.ClaOriginalCaptionType OriginalCaptionType, // the exact original caption type used for event tracking
    6: optional bool IsBurninCaption, // indicating if the video has burn_in caption
    7: optional bool CanTranslateRealtimeSkipTranslationLangCheck, // skip translation language check for real-time translation
}

struct CaptionStruct {
    1: required string Lang,                      // the language name from video arch subtitle management system
    2: required i64 LanguageId,                   // map<lang, language_id> is maintained by vcloud. https://bytedance.feishu.cn/docs/doccnL3XBAihnFwawShre7pjgyh
    3: required string Url,
    4: required i64 Expire,                       // CDN expiration time
    5: required string CaptionFormat = "webvtt",  // default "webvtt"
    6: required i64 ComplaintId,                  // used by tns to process complaints
    7: required bool IsAutoGenerated = true,
    8: optional i32 SubID,                        // video subtitle id from video arch
    9: optional string SubVersion,                // video subtitle version from video arch
    10: optional i64 ClaSubtitleId,               // video subtitle id from cla
    11: optional i64 TranslatorId,                // video subtitle translator id from cla
    12: optional string LanguageCode,             // the language code from lab which aligned with translation language
    13: optional bool IsOriginalCaption,          // determine if the caption is original or translation
    14: optional list<string> UrlList,            // the list of caption urls with major, backup and 302 redirect
    15: optional i64 CaptionLength;               // the length of caption, determined by len(record.Extra) when caption is generated
}

struct ClaStruct {
    1: required i32 HasOriginalAudio = 0,
    2: required i32 EnableAutoCaption = 0,
    3: optional OriginalLanguageStruct OriginalLanguageInfo,
    4: optional list<CaptionStruct> CaptionInfos,           // Video caption(cross language accsiibility) list which will dispatch suggested captions
    5: optional i64 CreatorEditedCaptionId,                 // caption id for creators to add caption back
    7: optional ClaCaptionPosition Positions,
    8: optional ClaCaptionAppearance Appearance,
    9: optional bool HideOriginalCaption,                   // Determine if the original caption needs to be hide
    10: optional enum.ClaCaptionsType CaptionsType,
    11: optional enum.ClaNoCaptionReason NoCaptionReason, // Why this item has no captions
}

struct ClaCaptionPosition {
    1: optional list<double> Vertical,
    2: optional double Horizontal,
}

struct ClaCaptionAppearance {
    1: optional string BackgroundColour, //bg_color
    2: optional double BackgroundCornerRadius, //bg_corner_radius
    3: optional i32 TextLabelAlignment,
    4: optional list<i32> TextLabelInsets,
    5: optional i32 CaptionTextSize,
    7: optional string CaptionTextColor,
    8: optional double CaptionTextStrokeWidth,
    9: optional string CaptionTextStrokeColor,
    10: optional bool ShouldShowCaptionIcon,
    11: optional bool ShouldShowInstructionText,
    12: optional i32 InstructionTextSize,
    13: optional string InstructionTextColor,
    14: optional double InstructionTextStrokeWidth,
    15: optional string InstructionTextStrokeColor,
    16: optional i32 ExpansionDirection,
    17: optional TextLabelInsetInfoStruct TextLabelInsetInfo,
    18: optional CaptionControlInfoStruct CaptionControlInfo,
}

struct CaptionControlInfoStruct {
    1: optional bool ShouldShowControlWhenCollapsed,
    2: optional bool ShouldShowControlWhenExpanded,
    3: optional bool ShouldShowCaptionOn,
    4: optional bool ShouldShowCaptionOff,
    5: optional bool TooltipHideEnabled,
}

struct TextLabelInsetInfoStruct {
    1: optional i32 Top,
    2: optional i32 Trailing,
    3: optional i32 Bottom,
    4: optional i32 Leading,
}

struct RecommendedCaptionStruct {
    1: required string Lang,
    2: required i64 Version,
    3: required i64 SubtitleId,
    4: optional i64 TtsVersion = 0, // deprecated
    5: optional list<string> DubbingVersions,
    6: optional i64 TranslatorId,
}

struct InteractPermissionStruct {
    1: required enum.InteractPermissionEnum Duet,                           // 0、1、2、3、4 --> able、disable、hide、disable4all、hide4all
    2: required enum.InteractPermissionEnum Stitch,                         // 0、1、2、3、4 --> able、disable、hide、disable4all、hide4all
    3: required enum.InteractPermissionEnum DuetPrivacySetting,             // video privacy setting control. 0,1,2 --> able,disable,hide
    4: required enum.InteractPermissionEnum StitchPrivacySetting,           // video privacy setting control. 0,1,2 --> able,disable,hide
    5: optional enum.InteractPermissionEnum Upvote,                         // 0、1、2、3、4 --> able、disable、hide、disable4all、hide4all
    6: optional enum.InteractPermissionEnum AllowAddingToStory,             // 0、1 --> able、disable
    7: optional InteractPermissionResultStruct AllowCreateSticker,
    8: optional InteractPermissionResultStruct AllowStorySwitchToPost,
}

struct InteractPermissionResultStruct {
    1: optional enum.InteractPermissionEnum Status, //0、1、2、3、4 --> able、disable、hide、disable4all、hide4all
    2: optional string DisableToast,
	3: optional string InteractionText, // / the interctive text if the interaction is allowed
}

struct BizAccountStruct {
    1: required list<string> PermissionList, // item BA permision list.
}

// Interaction Tag related
struct InteractionTagInfo {
   1: required enum.InteractionTagInterestLevel interestLevel,
   2: required string videoLabelText,
   3: list<i64> taggedUids,
   4: optional list<InteractionTagUserInfo> taggedUsers,
}

// Short User Info for TAG
struct InteractionTagUserInfo {
    1: optional string uid;
    2: optional string uniqueId;
    3: optional string nickname;
    4: optional UrlStruct avatar168;
    5: optional UrlStruct avatarThumb;
    6: optional enum.FollowStatusEnum followStatus;
    7: optional enum.FollowStatusEnum followerStatus;
    8: optional i32 isPrivateAccount;
    9: optional UserVerificationStruct verifyInfo;
    10: optional enum.InteractionTagInterestLevel userInterestLevel;
    11: optional bool isBusinessAccount;
    12: optional enum.TaggingBaInvitationStatus invitationStatus;
}

struct NoticeButtonStruct {
    1: optional string content;
    2: optional enum.NoticeButtonType type;
    3: optional string schema_url;
    4: optional list<enum.NoticeButtonActionType> actions;
}

struct DspStruct {
   1: optional i64 FullClipId;
   2: optional string FullClipAuthor;
   3: optional string FullClipTitle;
   4: optional i32 CollectStatus;
   5: optional MusicAvatarStruct PerformerDefaultAvatar;
   6: optional i64 MvId;
   7: optional bool IsShowEntrance;
}

struct UserPrivacyStruct {
   1: optional i32 UpvoteSetting;
   2: optional i32 ProfileViewHistory;
   3: optional i32 VideoViewHistory;
   4: optional i32 MentionStatus;
   5: optional i32 ShareToStorySetting;
   6: optional i32 CreateSticker;
   7: optional i32 PostToNearby;
}

struct TrendingBarForYouPageStruct {
    1: required UrlStruct IconUrl, //left icon url
    2: required string Display, //display text
    3: required UrlStruct ButtonIconUrl, //right button icon url
    4: required string Schema, //link
    5: optional i64 EventKeywordId,//event id
    6: optional string EventKeyword, //event name
    7: optional string EventTrackingParam, //client log param
}

struct MatchedFriendLabelStruct {
    1: optional string Text, // recommend text
    2: optional string TextColor,
    3: optional string BackgroundColor
    4: optional MutualStruct MutualInfo, // mutual relation info
    5: optional string SocialInfo, // for relation label bury
    6: optional string RecType,
    7: optional string FriendTypeStr,
    8: optional i64 RecommendType,
    9: optional string RelationTextKey,
}

struct ImagePostStruct {
    // 1: optional list<UrlStruct> Urls,
    // 2: optional UrlStruct PostCoverUrl,
    3: optional list<ImagePostInfo> Images,
    4: optional ImagePostInfo ImagePostCover,
    // The volume of the music, the value is from 0 to 1 inclusive
    5: optional double MusicVolume,
    6: optional string Title,
    // skip 7, deprecated
    // trace all image quality strategies
    8: optional string PhotoModeImageQualityInfo,
    100: optional string PostExtra,
}

struct ImagePostInfo {
    1: optional UrlStruct DisplayImage,
    2: optional UrlStruct OwnerWatermarkImage,
    3: optional UrlStruct UserWatermarkImage,
    4: optional UrlStruct Thumbnail,
    5: optional list<BitrateImagePostInfo> BitrateImages,
    6: optional UrlStruct DynamicImage,
}

struct BitrateImagePostInfo {
	1: optional string Name,
	2: optional UrlStruct BitrateImage,
}

struct LibraryMaterial {
    1: optional list<LibraryMaterialInfo> library_material_infos,
}

struct LibraryMaterialInfo {
    1: optional string id,
   	2: optional i64 start_time,
   	3: optional i64 end_time,
   	4: optional i64 material_provider,
    5: optional i64 material_type,
   	6: optional string name,
   	7: optional i64 used_count,
   	8: optional string cover,
}

struct ItemMusicInfo {
    1: optional i32 MusicBeginTimeInMs,
    2: optional i32 MusicEndTimeInMs,
    3: optional string OriginalVolume,
    4: optional string MusicVolume,
    5: optional MusicStruct AddedSoundMusicInfo
}

struct UserBadgeInfo {
    1: optional i64 Id,
    2: optional string Name,
    3: optional string Description,
    4: optional string Url,
    5: optional bool  ShouldShow,
}

struct MusicChartInfo{
    1: optional map<i64, i32> RankInfos,
}

struct UpvotePreloadStruct {
    1: optional bool NeedUpvoteInfo,
}

struct DurationHighPrecisionStruct {
    1: optional double DurationPrecision,
    2: optional double ShootDurationPrecision,
    3: optional double AuditionDurationPrecision,
    4: optional double VideoDurationPrecision,
}

struct ShareToStoryStruct {
    // The visibility of the forwarded story will change with the visibility of the original video,
    // Determines whether the forwarded story is visible or not
    1: optional i32 IsVisible,
    2: optional i64 OriginItemID,
    3: optional i32 ShareStoryPostType,
}

struct PaidContentInfo {
    1: optional bool is_paid_content,
    2: optional i64 paid_collection_id,
}

struct LocationInfo {
    1:optional LocationInfoDetail l0,
    2:optional LocationInfoDetail l1,
    3:optional LocationInfoDetail l2,
    4:optional LocationInfoDetail l3,
    50:optional string locate_method,
}

struct LocationInfoDetail {
    1:optional string code,
    2:optional double confidence,
}

struct NowPost {
	// 后置摄像头拍的图片，大图
    1:optional UrlStruct BackImage,
    // 后置摄像头拍的图片，缩略图
    2:optional UrlStruct BackImageThumbnail,
    // 前置摄像头拍的图片，大图
    3:optional UrlStruct FrontImage,
    // 前置摄像头拍的图片，缩略图
    4:optional UrlStruct FrontImageThumbnail,
    // 模糊图
    5:optional UrlStruct FuzzyImage,
    // 合成的图片，用于下载 ,ltr
    6:optional UrlStruct CompositeImageLtr,
    // 合成的图片，用于下载 ,rtl
    7:optional UrlStruct CompositeImageRtl,
    // 用户收到push的时间戳
    8:optional i64 LastPushedAtSec,
	// 前置摄像头模糊小图,缩略图，用于分享
    9:optional UrlStruct FuzzyFrontImageThumbnail,
    // 前置摄像头模糊小图,缩略图+logo，用于分享
    10:optional UrlStruct FuzzyFrontImageThumbnailWithLogo,
    // now posts visibility
    11:optional i32 NowStatus,
    // tiktok now incompatibility info
    12:optional NowIncompatibilityInfo IncompatibilityInfo,
	// now media type
	13:optional string NowMediaType,
    // 合成的图片(无水印)，用于下载,ltr
	14:optional UrlStruct CompositeImageLtrWithoutWatermark,
    // 合成的图片(无水印)，用于下载,rtl
	15:optional UrlStruct CompositeImageRtlWithoutWatermark,
	// 合成的图片(水印)，用于下载,ltr
	16:optional UrlStruct CompositeImageLtrWithWatermark,
	// 合成的图片(水印)，用于下载,rtl
	17:optional UrlStruct CompositeImageRtlWithWatermark,
	// 模糊图+now水印, ltr
	18:optional UrlStruct FuzzyImageLtrWithWatermark,
	// 模糊图+now水印, rtl
    19:optional UrlStruct FuzzyImageRtlWithWatermark,
    20:optional string CreateTimeInAuthorTimezone,
    // 分发的内容来源，主要是好友，热门以及关注的人
    21:optional enum.NowPostSource NowPostSource,
    // deprecated
    22: optional bool DisableInteraction,
    // 是否允许now post的交互，主要包括点赞，评论，举报等
    23:optional NowInteractionControl NowInteractionControl,
    // now 已读状态
	24:optional enum.NowViewState ViewState,
	// 是否包含聚合的now collection
    26:optional bool HasNowCollection,
    // now在fyp分发时的背景音乐id
    27:optional i64 BackgroundMusicID,
    // 字段分类Ref: https://bytedance.feishu.cn/docx/DgqEdOYvnoCUPWxsD2NcMD0rnSb
	28:optional NowPostAttributes Attributes,
	29:optional NowPostContentInfo ContentInfo,
	30:optional NowPostConsumptionInfo ConsumptionInfo,
	31: optional map<string, UrlStruct> ShareImage,
}

// Now Post的属性特征
struct NowPostAttributes {
	// 投稿过期时间
  1: optional i64 ExpiredAt,
  2: optional enum.NowPostCameraType NowPostCameraType, // Now投稿的拍摄类型
  3: optional bool IsOriginal, // 是否是直接拍摄投稿
}

// Now Post的素材内容
struct NowPostContentInfo {
}

// Now Post消费信息
struct NowPostConsumptionInfo{
    1:optional NowForcedVisibility ForcedVisibility,
    2:optional enum.NowBlurType BlurType, //模糊态展示样式
}

struct NowForcedVisibility {
    1:optional enum.ForcedVisibleState State,
    2:optional string Message,
}

struct NowInteractionControl {
    1:optional bool DisableLike,
    2:optional bool DisableComment,
    3:optional bool DisableReaction,
    4:optional enum.InteractionAction BlurCommentAction; //模糊态点击评论icon
    5:optional enum.InteractionAction BlurLikeAction; //模糊态点击点赞icon
}

struct NowIncompatibilityInfo {
    1:optional i32 Reason,
    // Localized string
    2:optional string Explain,
    3:optional NowButtonInfo ResolutionButton,
}

struct NowButtonInfo {
    // Localized string.
    1:optional string ButtonLabel,
    // Example: https://go.onelink.me/bIdt/409f077
    2:optional string RedirectUri,
}

struct BrandContentAccount {
    1: required i64 UserId,
    2: required string UserName
}

struct TTAInfo {
    1: optional bool IsVisible
}

struct PoiSubTag {
    1: required string                      Name,
    2: optional enum.PoiAnchorContentType   Type,
    3: optional i32                         Priority,
}

struct PoiAnchorInfo {
    1: required i64                     AnchorId,
    2: optional string                  Suffix,
    3: optional list<PoiSubTag>         SubTags,
    4: optional enum.PoiContentExpType  SubTagExpType,
    5: optional i32                     SubTagExpTime,
    6: optional bool                    HasSubArrow,
    7: optional string                  TrackInfo,
    8: optional list<enum.PoiHideType>  HideList,
}

struct PoiAddressInfo {
    1: optional string  CityName,
    2: optional string  CityCode,
    3: optional string  RegionCode,
    4: optional string  Lng,
    5: optional string  Lat,
}

struct PoiReviewConfig {
    1: optional bool ShowReviewTab,
}

struct ItemPoiStruct {
    1: required string          PoiName,
    2: required string          PoiId,
    3: optional string          PoiType,
    4: optional string          InfoSource,
    5: optional string          CollectInfo,
    6: optional bool            PoiMapKitCollect,
    7: optional i64             VideoCount,
    8: optional PoiAddressInfo  AddressInfo,
    9: optional PoiAnchorInfo   VideoAnchor,
    10: optional PoiAnchorInfo  CommentAnchor,
    11: optional bool           IsClaimed,
    12: optional string         TypeLevel,
    13: optional PoiReviewConfig PoiReviewConfig,
}

struct PodcastStruct {
    1: optional bool IsListenable,
    2: optional bool IsPodcast,
    3: optional enum.PodcastFollowDisplay FollowDisplay, // controls how to show follow button on audio feed
    4: optional string BackgroundColor,
    5: optional bool IsAudible,
    6: optional bool IsPodcastPreview,
    7: optional i64 FullEpisodeItemId,
    8: optional list<string> FullEpisodeAuthors,
    9: optional UrlStruct PodcastEpisodePlayAddr,
    10: optional UrlStruct PodcastEpisodeCoverImage,
    11: optional string PodcastEpisodeTitle,
    12: optional bool IsEpisodeBrandedContent,
    13: optional i64 PodcastEpisodeDurationMilliseconds,
    14: optional string PodcastFeedTitle,
    15: optional enum.PodcastTnsStatus PodcastEpisodeTnsStatus,
    16: optional enum.PodcastSharkStatus PodcastEpisodeSharkStatus,
}

struct UserNowPost {
    1:optional list<i64> NowPostIDs,
    2:optional i64 TotalCount,
    3:optional i64 NextCursor,
    4:optional i64 PreCursor,
    5:optional bool HasMoreAfter,
    6:optional bool HasMoreBefore,
    7:optional i64 Postion,
    8:optional enum.NowCollectionType CollectionType,
}

struct CCTemplateInfo {
    1: required string TemplateID, // template id in capcut
    2: optional string Desc,// desc of capcut template
    3: optional string AuthorName, // author name of capcut template
    4: optional i32 ClipCount, // count fo clips in capcut template video
    5: optional i32 DurationMilliseconds, // duration in millisconds for template video
    6: optional string RelatedMusicID, // related music id for template
}

struct UserNowPackStruct {
    1:optional enum.UserNowStatus UserNowStatus, // user now consumer status
    2:optional enum.NowAvatarBadgeStatus NowAvatarBadgeStatus, // now avatar badge status
}

struct UserAllStoryMetadataStruct {
    1:optional list<StoryLiteMetadata> StoryLiteMetadata,
    2:optional i64 LatestStoryViewTime, // item view version
    3:optional i64 UserStoryChangeTime, // user story change version
}

struct InteractionOtherInfo {
    1: optional bool IsTextStickerTranslatable; // identifies if the text stickers are translatable
    2: optional string TextStickerMajorLang; // the major language through all text stickers
    3: optional bool IsBurnInCaptionTranslatable; // identifies if the burn-in captions are translatable
}

struct TTMInfoStruct {
    1: optional enum.TTMProductType Product,
    2: optional TTMBrandStruct Brand,
    3: optional string VIPVerificationSchema,
}

struct TTMStoreLinkStruct {
    1: optional string Link,
    2: optional string Data,
}

struct TTMLinkStruct {
    1: optional string AppLink,
    2: optional string DeepLink,
    3: optional string DownloadLink,
    4: optional TTMStoreLinkStruct StoreLink,
}

struct TTMBrandStruct {
    1: optional string Entrance,
    2: optional TTMLinkStruct Link,
    3: optional string Title,
    4: optional string Subtitle,
    5: optional string ButtonText,
    6: optional bool Subscribed,
    7: optional enum.TTMUaSwitchStatus UaSwitchStatus,
}

struct TTMMusicInfoStruct {
    1: optional TTMTrackStruct Track,
}

struct TTMTrackStruct {
    1: optional i64 ID,
    2: optional string Name,
    3: optional UrlStruct CoverMedium,
    4: optional i64 Duration,
    5: optional string ArtistName,
    6: optional string LabelId,
    7: optional TTMTrackListenRightStruct ListenRight,
}

struct TTMTrackListenRightStruct {
    1: optional i32 Duration,
}

struct EditPostStruct {
    1: optional EditPostPermissionStruct permission,
    2: optional EditPostInfoStruct EditPostInfo,
}

struct EditPostPermissionStruct {
    1: list<EditPostBizPermissionStruct>   bizPermissions,
}

struct EditPostBizPermissionStruct {
    1: required enum.EditPostBiz            bizType,
    2: optional enum.EditPostControlStatus  bizStatus,
    3: optional enum.EditPostControlReason bizReason,
    4: optional EditPostVisibilityPermission visibilityPermission,
    // if the post is recyclable
    5: optional bool IsRecyclable,
}

struct EditPostVisibilityPermission {
    // edit permission of everyone visibility option
	6: optional enum.EditPostControlStatus VisibilityEveryone,
	// edit permission of followers visibility option
	7: optional enum.EditPostControlStatus VisibilityFollowers,
	// edit permission of friends visibility option
	8: optional enum.EditPostControlStatus VisibilityFriends,
	// edit permission of only-you visibility option
	9: optional enum.EditPostControlStatus VisibilityOnlyYou,
	// edit permission of subscriber-only visibility option
	10: optional enum.EditPostControlStatus VisibilitySubOnly,
	// edit permission of available-for-ads visibility option
	11: optional enum.EditPostControlStatus VisibilityAvailableForAds,
}

struct TextEditStruct {
	// Descriptions on tree:
	// ContentModel.StandardBiz.EditPost.EditPostInfo.TextEditInfo.Description - to be discussed
	2: optional DescriptionStruct Description;
	// Descriptions on tree:
	// ContentModel.StandardBiz.EditPost.EditPostInfo.TextEditInfo.IsDescNotChanged - if post's description is submitted for change
	3: optional bool IsDescNotChanged;
}

struct DescriptionStruct {
	// Descriptions on tree:
	// ContentModel.StandardBiz.EditPost.EditPostInfo.TextEditInfo.Description.Visibility - visibility
	1: optional i32 Visibility;
	// Descriptions on tree:
	// ContentModel.StandardBiz.EditPost.EditPostInfo.TextEditInfo.Description.Title - title
	2: optional string Title;
	3: optional string Text;
	4: optional string TextLanguage;
	5: optional list<TextExtraStruct> TextExtra;
	// content caption
	6: optional string ContentDesc;
	// extra info for content desc
	7: optional list<TextExtraStruct> ContentDescExtra;
	// Identifies if the description content is translatable.
	8: optional bool Translatable;
	// the markup_text and text_extra field generated by app client directly
	9: optional string ClientText;
	// CLA related, identifies if the image post title is translatable.
	10: optional bool IsTitleTranslatable;
	// CLA related, identifies the language of image post title.
	11: optional string TitleLanguage;
}

struct CoverEditStruct {
	1: optional UrlStruct Cover;
}

struct EditPostInfoStruct {
    1: optional TextEditStruct TextEditInfo;
	2: optional CoverEditStruct CoverEditInfo;
}

// ref: https://bytedance.feishu.cn/docx/SL1dd13xIocttLxgCAccLTUYnJe
struct TextToSpeechStruct {
    1: optional list<string> TtsVoiceIds; // tts voice ids have been applied to the video
    2: optional list<string> ReferenceTtsVoiceIds; // referenced tts voice ids have been used to create the video
}

// ref: https://bytedance.feishu.cn/docx/SL1dd13xIocttLxgCAccLTUYnJe
struct VoiceChangeFilterStruct {
    1: optional list<string> VoiceFilterIds; // voice change filter ids have been applied to the video
    2: optional list<string> ReferenceVoiceFilterIds; // referenced voice change filiter ids have been used to create the video
}

// ref: https://bytedance.feishu.cn/docx/Ohs4d2OAAoasTJxEVaacJZAankf
struct AnimatedImageStruct {
    1: optional i32 Type, // 1: livephoto 2: gif
    2: optional i32 Effect, // 1: live 2: rotation
}

struct EmojiRecommend {
    1: required string emoji
    2: optional i32 score
}

// ref https://bytedance.feishu.cn/docx/DofkdfKzjopdT3xHmeNcWffAn2e
struct CommentConfigStruct {
   1: optional string ZeroIconText, // icon text when comment count is zero
   2: optional string ZeroInputBoxText, // input box text when comment count is zero
   3: optional string NonZeroInputBoxText, // input box text when comment count is zero
   4: optional string EmptyListText, // box text when comment count is zero
   5: optional list<EmojiRecommend> EmojiRecommendList // recommend emoji list
}

struct SchedulePostStruct {
    1: required i64 CreateTime,
    2: required i64 ScheduleTime,
    3: required i32 OriUserStatus, // original status chose by user
    4: required bool Finish,
}

struct SerializableItemStruct {
    1: optional enum.SerializableItemType SerializableType,
    2: optional string RawData,
}

struct MufCommentInfoStruct {
    1: optional i64 CommentId,
    2: optional string NickName,
    3: optional UrlStruct AvatarThumbUrl,
    4: optional UrlStruct AvatarMediumUrl,
}

struct DanmakuInfoStruct {
    1: optional bool SupportDanmaku,
    2: optional bool HasDanmaku,
}

struct AddYoursStickerStruct {
    1: optional i64 TopicID,  // both creation/consumption needed; add_yours topic id; set empty if it's new topic
    2: optional i64 FromItemID,  // creation only, set empty if it's new topic
    3: optional string Text,  // both creation/consumption needed; topic text
    4: optional list<AddYoursAvatar> UserAvatars,  // consumption only
    5: optional i64 VideosCount,  // consumption only; total videos
    6: optional bool FromQuestion,  // creation only;
    7: optional list<i64> AddYoursInvitees,  // creation only;
    8: optional bool ViewerIsInvited,  // consumption only;
    9: optional bool EoyCampaign,
    10: optional string EoyCampaignSchema,
    11: optional i32 TopicType,
}

struct AddYoursAvatar {
    1: optional i64 Uid,
    2: optional UrlStruct UserAvatar,
}

// MinT Operator boost
struct OperatorBoostStruct {
    1: optional i64 TargetVv,
    2: optional i64 EndTime,
    3: optional string Region,
}

struct FrequencyConfig {
    1: required enum.NoticeActionEnum user_action, // 触发场景
    2: optional NoticeFrequencyCondition condition, // 触发条件
    3: required NoticeBehavior behavior,  // 行为
}

struct NoticeFrequencyCondition{
    1: optional i64 threshold,
}

struct NoticeBehavior{
    1: required enum.NoticeBehaviorActionEnum action,
    2: optional NoticeBehaviorParam params,
}

struct NoticeBehaviorParam{
    1: optional enum.NoticeOperationTypeEnum op_type,
    2: optional string extra,
}

// prd: https://bytedance.feishu.cn/docx/BWDidag0Ooco3Jx5jOicjZcTnze
struct UserUnreadItem {
    1: optional list<i64> ItemIds,
}

struct AigcInfoStruct {
    1: required enum.AigcLabelType AigcLabelType,
}

struct RelatedLiveStruct {
    1: required string content,
    2: required string related_tag,
    3: required string tag_name,
}

struct TrendingRecallInfoStruct {
    1: optional enum.TrendingProductRecallType TrendingProductRecallType,
}

struct AddYoursRecommendationStruct {
    1: optional i64 topic_id,
    2: optional string topic_text,
}

struct AddToStoryStruct {
    1: optional enum.ShareStoryStatusEnum ShareStoryStatus,
}

struct OriginalAudioStruct {
    1: optional UrlStruct PlayUrl, // original audio play url
    2: optional string Vid,         // original audio vid
    3: optional double Volume,      // original audio volume
}

struct MemeSongStruct {
   // is a sound a meme song
   1: optional bool IsMemeSong,
   // meme song's style
   2: optional string MemeSongStyle,
}

struct FullSongStruct {
    // id of full song
    1: optional i64 FullSongId,
    // duration of full song
    2: optional i32 FullSongDuration,
    // precision duration of full song
    3: optional double FullSongDurationPrecision,
    // shoot duration of full song
    4: optional i32 FullSongShootDuration,
    // shoot duration precision of full song
    5: optional double FullSongShootDurationPrecision,
    // full play url for full song
    6: optional UrlStruct FullSongPlayUrl,
    // strong beat url for full song
    7: optional UrlStruct FullSongStrongBeatUrl,
}

struct FusedMusicStruct {
    1: optional i64 ID,
    2: optional i32 Index,
    3: optional i32 StartTimeInMs,
    4: optional i32 EndTimeInMs,
}

struct MusicFusionStruct {
    1: optional list<FusedMusicStruct> FusedMusicInfos,
    2: optional bool IsSmartExtended,
}

struct AvatarMetaInfo {
    1: optional enum.AvatarSourceEnum AvatarSource,
    2: optional enum.AvatarDefaultShowEnum AvatarChoice,
    3: optional i32 SocialAvatarBackgroundTag,
    4: optional i32 SocialAvatarExpressionTag,
    5: optional i64 SocialAvatarId,
}

struct AvatarCategoryMetaInfo {
    1: optional enum.AvatarCategory AvatarCategory,
    2: optional AvatarMetaInfo AvatarMetaInfo,
}

struct AvatarSizeUrlStruct {
    1: optional UrlStruct avatar_larger,
    2: optional UrlStruct avatar_medium,
    3: optional UrlStruct avatar_thumb,
    4: optional UrlStruct avatar_168x168,
    5: optional UrlStruct avatar_300x300,
}
