namespace go tiktok.content.standard_biz

include "common.thrift"
include "content_base.thrift"
include "enum.thrift"

struct ABRollInfo {
	// Descriptions on tree:
	// ContentModel.StandardBiz.ABRoll.ABRollInfoA.MediaInfoType - Type of media, such as video or image
	1: optional enum.MediaInfoType MediaInfoType;
	// Descriptions on tree:
	// ContentModel.StandardBiz.ABRoll.ABRollInfoA.Index - Index of corresponding image in ImagePostInfo.Images
	2: optional i32 Index;
}

struct ABRollStruct {
	1: optional ABRollInfo ABRollInfoA;
	2: optional ABRollInfo ABRollInfoB;
}

struct ActivityCommerceStruct {
	// red envelope type 1: gesture red envelope ，2: KOL pendant
	1: optional enum.CommerceActivityTypeEnum ActType;
	// openurl
	2: optional string JumpOpenUrl;
	// H5url
	3: optional string JumpWebUrl;
	4: optional string Title;
	5: optional common.UrlStruct Image;
	6: optional i64 StartTime;
	7: optional i64 EndTime;
	8: optional list<common.TimeRangeInDoublePrecision> TimeRanges;
	// Third party monitoring url
	9: optional string TrackUrl;
	// Third party click monitoring url
	10: optional string ClickTrackUrl;
}

struct ActivityTrilateralCooperationStruct {
	1: optional string Desc;
	2: optional string Title;
	3: optional string JumpUrl;
	4: optional string IconUrl;
	5: optional bool IsTask;
	6: optional i32 SwitchType;
	7: optional string EntranceUrl;
}

struct ActivityStruct {
	1: optional ActivityCommerceStruct ActivityPendantInfo;
	2: optional ActivityCommerceStruct GestureRedPacketInfo;
	3: optional ActivityTrilateralCooperationStruct TrilateralCooperationInfo;
}

struct AddToStoryStruct {
	1: optional enum.ShareStoryStatusEnum ShareStoryStatus;
}

struct AddYoursAvatar {
	1: optional i64 Uid;
	2: optional common.UrlStruct UserAvatar;
}

struct AddYoursInfo {
	// Descriptions on tree:
	// ContentModel.StandardBiz.AddYoursInfo.AddYoursTrendVideoSourceEnum - addyours trend video source
	1: optional enum.AddYoursTrendVideoSourceEnum AddYoursTrendVideoSourceEnum;
}

struct AddYoursRecommendationStruct {
	1: optional i64 TopicId;
	2: optional string TopicText;
	// Descriptions on tree:
	// ContentModel.StandardBiz.AddYoursRecommendationInfo.TopicType - addyours topic type
	3: optional enum.AddYoursTopicTypeEnum TopicType;
	// Descriptions on tree:
	// ContentModel.StandardBiz.AddYoursRecommendationInfo.VideoSource - addyours video source enum
	4: optional enum.AddYoursTrendVideoSourceEnum VideoSource;
}

struct AddYoursStickerStruct {
	// both creation/consumption needed; add_yours topic id; set empty if it's new topic
	1: optional i64 TopicID;
	// creation only, set empty if it's new topic
	2: optional i64 FromItemID;
	// both creation/consumption needed; topic text
	3: optional string Text;
	// consumption only
	4: optional list<AddYoursAvatar> UserAvatars;
	// consumption only; total videos
	5: optional i64 VideosCount;
	// creation only;
	6: optional bool FromQuestion;
	// creation only;
	7: optional list<i64> AddYoursInvitees;
	// consumption only;
	8: optional bool ViewerIsInvited;
	// Descriptions on tree:
	// ContentModel.StandardBiz.Sticker.InteractionStickers.AddYoursSticker.EoyCampaign - is eoy campaign
	9: optional bool EoyCampaign;
	// Descriptions on tree:
	// ContentModel.StandardBiz.Sticker.InteractionStickers.AddYoursSticker.EoyCampaignSchema - eoy campaign schema
	10: optional string EoyCampaignSchema;
	// topic type
	11: optional i32 TopicType;
}

struct AigcInfoStruct {
	1: optional enum.AigcLabelType AigcLabelType;
	// Descriptions on tree:
	// ContentModel.StandardBiz.AigcInfo.CreatedByAI - Whether this aweme struct is created by AI
	2: optional bool CreatedByAI;
}

struct AnchorActionStruct {
	1: optional common.UrlStruct Icon;
	2: optional string Schema;
	3: optional enum.AnchorActionType ActionType;
}

struct AnchorCommonStruct {
	1: optional string Id;
	2: optional enum.AnchorType Type;
	3: optional string Keyword;
	4: optional string Url;
	5: optional common.UrlStruct Icon;
	6: optional string Schema;
	7: optional string Language;
	8: optional string Extra;
	9: optional string DeepLink;
	10: optional string UniversalLink;
	11: optional enum.AnchorGeneralType GeneralType;
	12: optional string LogExtra;
	13: optional string Description;
	14: optional common.UrlStruct Thumbnail;
	15: optional list<AnchorActionStruct> Actions;
	16: optional map<string,string> ExtraInfo;
	17: optional bool IsShootingAllow;
	18: optional string ComponentKey;
	// feed 上的第二行文案
	19: optional string Caption;
}

struct AnchorStrategyStruct {
	1: optional bool SupportSingleAnchor;
	2: optional bool SupportMultiAnchors;
	3: optional bool RemoveMvInfo;
}

// ref: https://bytedance.feishu.cn/docx/Ohs4d2OAAoasTJxEVaacJZAankf
struct AnimatedImageUploadStruct {
	// 1: livephoto 2: gif
	1: optional i32 Type;
	// 1: live 2: rotation
	2: optional i32 Effect;
}

struct AppleMusicToken {
	// Descriptions on tree:
	// ContentModel.StandardBiz.TT2DSPAlbumInfo.Token.AppleMusicToken.DeveloperToken - apple music developer token
	1: optional string DeveloperToken;
}

struct ArtistStruct {
	1: optional list<i64> PickedUserIDs;
}

struct AttributionLinkStickerStruct {
	// Descriptions on tree:
	// ContentModel.StandardBiz.Sticker.InteractionStickers.AttributionLinkSticker.Scenario - The scenario of the attribution link sticker
	1: optional enum.AttributionLinkScenario Scenario;
	// Descriptions on tree:
	// ContentModel.StandardBiz.Sticker.InteractionStickers.AttributionLinkSticker.Title - The text show on the attribution link sticker
	2: optional string Title;
	// Descriptions on tree:
	// ContentModel.StandardBiz.Sticker.InteractionStickers.AttributionLinkSticker.Url - The url that user will jump to after clicking
	3: optional string Url;
	// Descriptions on tree:
	// ContentModel.StandardBiz.Sticker.InteractionStickers.AttributionLinkSticker.ShareFormat - the share format of the link sharing functionality
	4: optional enum.AttributionLinkShareFormat ShareFormat;
}

struct AutoCaptionDetailStruct {
	1: optional string Language;
	2: optional common.UrlStruct Url;
}

struct AutoCaptionPositionStruct {
	1: optional list<double> Vertical;
	2: optional double Horizontal;
}

struct AwemeLabelStruct {
	1: optional common.UrlStruct LabelUrl;
	2: optional i32 LabelType;
}

struct BaAnchorInfo {
	// Descriptions on tree:
	// ContentModel.StandardBiz.BaInfo.BaAnchorInfos.AnchorType - BA attach anchor type
	1: optional i64 AnchorType;
	// Descriptions on tree:
	// ContentModel.StandardBiz.BaInfo.BaAnchorInfos.AnchorSource - BA attach anchor source
	2: optional i64 AnchorSource;
}

struct BaInfo {
	1: optional list<BaAnchorInfo> BaAnchorInfos;
}

struct BannerAction {
	// 跳转 schema，业务方实现
	1: optional string Schema;
	// 选择反馈类使用（二选一）
	2: optional string Text;
	3: optional common.UrlStruct Deprecated;
	// 适配跳转按钮上需要展示图标的情况（二选一）
	4: optional string Icon;
}

struct BannerKey {
	// FCP 平台的唯一key
	1: optional string ComponentKey;
	// 业务的唯一标识
	2: optional string BusinessId;
}

struct BannerTailAction {
	1: optional enum.BannerActionType Type;
	2: optional list<BannerAction> Actions;
}

struct BannerText {
	1: optional string Body;
	2: optional string TextColor;
	// 跳转 schema
	3: optional string Schema;
	4: optional enum.BannerTextActionType ActionType;
}

struct BannerContent {
	// 实际内容
	1: optional list<BannerText> Texts;
	// 一行或两行
	2: optional i32 MaxLines;
}

struct BannerStandardUI {
	// 文本
	1: optional BannerContent Content;
	2: optional string HeadIcon;
	// 交互跳转，适配多个跳转的情况。
	3: optional BannerTailAction TailAction;
}

// banner 统一埋点
struct BannerTracerInfo {
}

struct BannerStruct {
	// 一个 banner 的唯一标识
	1: optional BannerKey Key;
	// banner 标准化 UI
	2: optional BannerStandardUI StandardUI;
	// 业务方 banner 透传字段，但每种类型的 Banner 必须固定结构体类型
	3: optional string CustomizedInfo;
	// banner 统一埋点字段
	4: optional BannerTracerInfo TracerInfo;
}

struct BatchPostStruct {
	// ID of each batch post
	1: optional string BatchId;
	// post position of the batch post
	2: optional i32 BatchIndex;
}

struct BizAccountStruct {
	// item BA permision list.
	1: optional list<string> PermissionList;
}

struct BodydanceStruct {
	// bodydance得分, 猜测废弃了：Pack和Feeds中均有赋值，但是item中应该不用了。
	1: optional i64 BodydanceScore;
}

struct BottomBarStruct {
	1: optional enum.BottomBarTypeEnum Type;
	2: optional string Content;
	3: optional common.UrlStruct Icon;
}

struct BrandContentAccountStruct {
	1: optional i64 UserId;
	2: optional string UserName;
}

struct C2paInfo {
	// Descriptions on tree:
	// ContentModel.StandardBiz.C2paInfo.FirstSrc - Primary c2pa valid source
	1: optional string FirstSrc;
	// Descriptions on tree:
	// ContentModel.StandardBiz.C2paInfo.LastSrc - Last c2pa valid source
	2: optional string LastSrc;
	// Descriptions on tree:
	// ContentModel.StandardBiz.C2paInfo.DedupSrc - Deduplicated platform list
	3: optional string DedupSrc;
	// Descriptions on tree:
	// ContentModel.StandardBiz.C2paInfo.FirstAigcSrc - First valid AIGC source type
	4: optional string FirstAigcSrc;
	// Descriptions on tree:
	// ContentModel.StandardBiz.C2paInfo.LastAigcSrc - Last valid AIGC source type
	5: optional string LastAigcSrc;
	// Descriptions on tree:
	// ContentModel.StandardBiz.C2paInfo.DedupErr - Valid c2pa reading error code
	6: optional string DedupErr;
	// Descriptions on tree:
	// ContentModel.StandardBiz.C2paInfo.AigcSrc - Source of content creation
	7: optional string AigcSrc;
	// Descriptions on tree:
	// ContentModel.StandardBiz.C2paInfo.IsTiktok - Whether aigc is from tiktok
	8: optional bool IsTiktok;
	// Descriptions on tree:
	// ContentModel.StandardBiz.C2paInfo.IsCapcut - Whether aigc is from capcut
	9: optional bool IsCapcut;
	// Descriptions on tree:
	// ContentModel.StandardBiz.C2paInfo.TotalSrc - Total valid c2pa source type
	10: optional i64 TotalSrc;
	// Descriptions on tree:
	// ContentModel.StandardBiz.C2paInfo.TotalErr - Total number of error codes
	11: optional i64 TotalErr;
	// Descriptions on tree:
	// ContentModel.StandardBiz.C2paInfo.TotalImg - Total number of valid c2pa image files
	12: optional i64 TotalImg;
	// Descriptions on tree:
	// ContentModel.StandardBiz.C2paInfo.TotalVid - Total number of valid c2pa video files
	13: optional i64 TotalVid;
	// Descriptions on tree:
	// ContentModel.StandardBiz.C2paInfo.UploadDur - Total length of c2pa metadata
	14: optional double UploadDur;
}

struct CapCutTemplateStruct {
	// template id in capcut
	1: optional string TemplateId;
	// desc of capcut template
	2: optional string Desc;
	// author name of capcut template
	3: optional string AuthorName;
	// count fo clips in capcut template video
	4: optional i32 ClipCount;
	// duration in millisconds for template video
	5: optional i32 DurationMilliseconds;
	// related music id for template
	6: optional string RelatedMusicId;
}

struct CaptionControlInfoStruct {
	1: optional bool ShouldShowControlWhenCollapsed;
	2: optional bool ShouldShowControlWhenExpanded;
	3: optional bool ShouldShowCaptionOn;
	4: optional bool ShouldShowCaptionOff;
	5: optional bool TooltipHideEnabled;
}

struct CelebrationItemStruct {
	// Descriptions on tree:
	// ContentModel.StandardBiz.CelebrationInfo.CelebrationType - celebration type
	1: optional enum.CelebrationType CelebrationType;
}

// ------ 功能信息 ------
struct ChallengeStruct {
	1: optional list<i64> ChallengeIds;
}

struct ChapterDetail {
	// Descriptions on tree:
	// ContentModel.StandardBiz.Chapter.ChapterDetails.Desc - Text description of the chapter
	1: optional string Desc;
	// Descriptions on tree:
	// ContentModel.StandardBiz.Chapter.ChapterDetails.StartInMS - Start time in milliseconds of the chapter
	2: optional i32 StartInMS;
}

struct Chapter {
	// Descriptions on tree:
	// ContentModel.StandardBiz.Chapter.ChapterDetails - List of each chapter object
	1: optional list<ChapterDetail> ChapterDetails;
}

struct ClientAIExtraInfo {
	// Descriptions on tree:
	// ContentModel.StandardBiz.ClinetAIExtraInfo.UserAffinityScore - 作者和消费者的亲密度
	1: optional double UserAffinityScore;
}

struct CoinIncentiveVideoInfoStruct {
	// Descriptions on tree:
	// ContentModel.StandardBiz.CoinIncentiveVideoInfo.IncentiveVideoType - using for tag the specific video type and show special share button
	1: optional string IncentiveVideoType;
	// Descriptions on tree:
	// ContentModel.StandardBiz.CoinIncentiveVideoInfo.TagExpire - using to control the tag expire time
	2: optional i64 TagExpire;
}

struct CollabUserStruct {
	// Descriptions on tree:
	// ContentModel.StandardBiz.CollabInfo.CollabUsers.Id - userID
	1: optional i64 Id;
	// Descriptions on tree:
	// ContentModel.StandardBiz.CollabInfo.CollabUsers.Status - status for collaboration
	2: optional enum.CollabStatusEnum Status;
	// Descriptions on tree:
	// ContentModel.StandardBiz.CollabInfo.CollabUsers.AuthorCollaboratorBlockStatus - the blocking relationship between author and the collaborator
	3: optional enum.BlockStatusEnum AuthorCollaboratorBlockStatus;
}

struct CollabInfoStruct {
	// Descriptions on tree:
	// ContentModel.StandardBiz.CollabInfo.IsCollab - whether this item is collab or not
	1: optional bool IsCollab;
	2: optional list<CollabUserStruct> CollabUsers;
	// Descriptions on tree:
	// ContentModel.StandardBiz.CollabInfo.ShouldDisplayTag - whether to show collab tag or not
	3: optional bool ShouldDisplayTag;
}

struct CollectStruct {
}

struct CommentFilterStrategyStruct {
	// decide if filter unfriendly user comment https://bytedance.feishu.cn/docx/PtUVdVcPNoch3IxldwGc7A60nrf
	1: optional bool FilterUnfriendlyUserComments;
	// decide if filter unfriendly user comment https://bytedance.feishu.cn/docx/PtUVdVcPNoch3IxldwGc7A60nrf
	2: optional bool FilterStrangerComments;
}

struct CommentPostStickerStruct {
	// aweme_id of the original video
	1: optional string OriginalItemId;
	// original commment id
	2: optional string OriginalCommentId;
}

struct CommentPostStruct {
	1: optional i64 OriginItemID;
	2: optional i64 OriginCommentID;
	3: optional i32 IsVisible;
	4: optional bool IsCommentPostVideo;
}

struct CommentPreload {
	// Descriptions on tree:
	// ContentModel.StandardBiz.Interaction.Comment.CommentConfig.CommentPreload.Type - 0:don not reload; not 0: reload and request args with reload
	1: optional i32 Type;
	// Descriptions on tree:
	// ContentModel.StandardBiz.Interaction.Comment.CommentConfig.CommentPreload.Preds - example: "{\"click_comment\":0.1}
	2: optional string Preds;
}

struct CommentTopBarStruct {
	1: optional i64 ID;
	2: optional string Type;
	3: optional string SubType;
	4: optional i64 Exposures;
	5: optional i64 StartTime;
	6: optional i64 EndTime;
	7: optional common.UrlStruct IconUrl;
	8: optional string Keywords;
	9: optional string Description;
	10: optional string Schema;
	11: optional string Url;
	12: optional i64 Priority;
	13: optional string Language;
	14: optional string Extra;
	15: optional string LogExtra;
	16: optional string DeepLink;
	17: optional string UniversalLink;
	18: optional common.UrlStruct Thumbnail;
}

struct ComponentEventTrackStruct {
	// Descriptions on tree:
	// ContentModel.StandardBiz.StandardComponentInfo.ComponentEventTrackInfo.ComponentEventTrackKey - 埋点key
	1: optional string ComponentEventTrackKey;
	// Descriptions on tree:
	// ContentModel.StandardBiz.StandardComponentInfo.ComponentEventTrackInfo.ComponentEventTrackValue - 埋点value
	2: optional list<string> ComponentEventTrackValue;
}

struct ComponentShowInfoStruct {
	// Descriptions on tree:
	// ContentModel.StandardBiz.RealTimeFcpConfig.RealtimeComponentConfig.ShowInfoResetComponents.ComponentKey - 组件key
	1: optional string ComponentKey;
	// Descriptions on tree:
	// ContentModel.StandardBiz.RealTimeFcpConfig.RealtimeComponentConfig.ShowInfoResetComponents.ShowStrategy - 组件展示策略
	2: optional enum.ShowStrategyEnum ShowStrategy;
}

struct ConfigDataItemLikeEggStruct {
	1: optional string MaterialUrl;
	2: optional string FileType;
}

struct ConfigDataStickerPendantStruct {
	1: optional i32 StickerType;
	2: optional string Link;
	3: optional string Title;
	4: optional string StickerId;
	5: optional common.UrlStruct IconUrl;
	6: optional string OpenUrl;
}

struct CommerceConfigStruct {
	1: optional string Id;
	2: optional i32 Type;
	3: optional string Data;
	4: optional i32 TargetType;
	5: optional string TargetId;
	6: optional ConfigDataItemLikeEggStruct ItemLikeEgg;
	7: optional ConfigDataStickerPendantStruct StickerPendant;
}

struct ContentCheckInfo {
	// Descriptions on tree:
	// ContentModel.StandardBiz.ContentCheckInfo.ContentCheckStatus - This field is used to reflect the result of Content Check (pre publish content moderation result)
	1: optional string ContentCheckStatus;
}

struct CoverEditStruct {
	1: optional common.UrlStruct Cover;
}

struct CreationInfoStruct {
	// Descriptions on tree:
	// ContentModel.StandardBiz.CreationInfo.CreationUsedFunctions - All function names used  during aweme struct creation
	1: optional list<string> CreationUsedFunctions;
	// Descriptions on tree:
	// ContentModel.StandardBiz.CreationInfo.TimePortal - Time portal original post timestamp
	2: optional i64 TimePortal;
}

struct CreatorAnalyticsStruct {
	// refer enum, use to enable/disable video insight H5 page when creator views from profile page, see: https://bytedance.us.feishu.cn/docx/ILdodzikloeb0bxYZyduVYngsRo
	1: optional enum.InsightStatusEnum VideoInsightStatus;
	// Descriptions on tree:
	// ContentModel.StandardBiz.CreatorAnalytics.ItemAnalyticsDataStatus - item analytics data status
	2: optional enum.AnalyticsDataStatus ItemAnalyticsDataStatus;
	// Descriptions on tree:
	// ContentModel.StandardBiz.CreatorAnalytics.CreatorAnalyticsDataStatus - creator analytics data status
	3: optional enum.AnalyticsDataStatus CreatorAnalyticsDataStatus;
	// Descriptions on tree:
	// ContentModel.StandardBiz.CreatorAnalytics.ShowAnalyticsDataEntrance - if show the analytics data entrance
	4: optional bool ShowAnalyticsDataEntrance;
}

struct CreatorPlaylistStruct {
	// playlist id
	1: optional i64 MixId;
	// creator
	2: optional i64 UserId;
	// name of playlist
	3: optional string Name;
	// playlist status
	4: optional i32 Status;
	// review status
	5: optional i32 Review;
	// mix name in moderation review, can only be seen by the owner
	6: optional string NameInReview;
	// whether the current playlist is using default name
	7: optional bool IsDefaultName;
	// index of the video in this mix
	8: optional i64 Index;
	// total number of item
	9: optional i64 ItemTotal;
}

struct DSPEntityInfoStruct {
	// Descriptions on tree:
	// ContentModel.StandardBiz.TT2DSPAlbumInfo.PreSaveInfo.DspEntityInfos.DspEntityType - dsp entity type
	1: optional string DspEntityType;
	// Descriptions on tree:
	// ContentModel.StandardBiz.TT2DSPAlbumInfo.PreSaveInfo.DspEntityInfos.DspEntityID - dsp entity id
	2: optional string DspEntityID;
	// Descriptions on tree:
	// ContentModel.StandardBiz.TT2DSPAlbumInfo.PreSaveInfo.DspEntityInfos.Platform - dsp platform enum
	3: optional enum.DspPlatform Platform;
	// Descriptions on tree:
	// ContentModel.StandardBiz.TT2DSPAlbumInfo.PreSaveInfo.DspEntityInfos.SourceEntityType - tiktok entity type
	4: optional string SourceEntityType;
	// Descriptions on tree:
	// ContentModel.StandardBiz.TT2DSPAlbumInfo.PreSaveInfo.DspEntityInfos.SourceEntityID - tiktok entity id
	5: optional string SourceEntityID;
}

struct DanmakuStruct {
	1: optional bool SupportDanmaku;
	2: optional bool HasDanmaku;
}

struct DarkPostInfoStruct {
	// extra.dark_post_info.Status
	1: optional i32 Status;
	// extra.dark_post_info.Source
	2: optional i32 Source;
}

struct ADInfoStruct {
	// 广告来源
	1: optional enum.AdSource AdSource;
	// 广告授权状态
	2: optional enum.AdAuthStatus AdAuthStatus;
	3: optional string AdTitle;
	4: optional string AdAvatarUri;
	// Are advertisers allowed to promote
	5: optional bool AdvPromotable;
	// Has the video ever been invited to bid for MT advertising
	6: optional bool AuctionAdInvited;
	// ban some action by commerce reason
	7: optional i32 AdBanMask;
	// 增加下发评论开关的逻辑
	8: optional bool CanCommentForAd;
	// 视频是否被设置评论过滤关键词
	9: optional bool WithCommentFilterWords;
	// 无版权的广告音乐
	10: optional bool WithPromotionalMusic;
	// extra.dark_post_info
	11: optional DarkPostInfoStruct DarkPostInfo;
	12: optional enum.DiggShowScene DiggShowScene;
}

struct DescendantStruct {
	1: optional list<string> SiblingDescs;
	2: optional list<string> SiblingNames;
}

struct DownloadStruct {
}

struct DspAuthToken {
	1: optional AppleMusicToken AppleMusicToken;
}

struct DuetStruct {
	// The original video item ID duetted from
	1: optional i64 OriginalItemId;
	// The duet layout, e.g. new_left
	2: optional i32 DuetLayout;
}

struct DuetWithMeStickerStruct {
	// auto-turn on/off mic for duet
	1: optional bool MicPermissionOn;
	// content of duet with me sticker like "Duet with me"
	2: optional string StickerContent;
}

struct EditPostVisibilityPermission {
	// Descriptions on tree:
	// ContentModel.StandardBiz.EditPost.Permission.BizPermissions.VisibilityPermission.VisibilityEveryone - edit permission of everyone visibility option
	6: optional enum.EditPostControlStatus VisibilityEveryone;
	// Descriptions on tree:
	// ContentModel.StandardBiz.EditPost.Permission.BizPermissions.VisibilityPermission.VisibilityFollowers - edit permission of followers visibility option
	7: optional enum.EditPostControlStatus VisibilityFollowers;
	// Descriptions on tree:
	// ContentModel.StandardBiz.EditPost.Permission.BizPermissions.VisibilityPermission.VisibilityFriends - edit permission of friends visibility option
	8: optional enum.EditPostControlStatus VisibilityFriends;
	// Descriptions on tree:
	// ContentModel.StandardBiz.EditPost.Permission.BizPermissions.VisibilityPermission.VisibilityOnlyYou - edit permission of only-you visibility option
	9: optional enum.EditPostControlStatus VisibilityOnlyYou;
	// Descriptions on tree:
	// ContentModel.StandardBiz.EditPost.Permission.BizPermissions.VisibilityPermission.VisibilitySubOnly - edit permission of subscriber-only visibility option
	10: optional enum.EditPostControlStatus VisibilitySubOnly;
	// Descriptions on tree:
	// ContentModel.StandardBiz.EditPost.Permission.BizPermissions.VisibilityPermission.VisibilityAvailableForAds - edit permission of available-for-ads visibility option
	11: optional enum.EditPostControlStatus VisibilityAvailableForAds;
}

struct EditPostBizPermissionStruct {
	1: optional enum.EditPostBiz BizType;
	2: optional enum.EditPostControlStatus BizStatus;
	// Descriptions on tree:
	// ContentModel.StandardBiz.EditPost.Permission.BizPermissions.BizReason - edit-post module entry display reason
	3: optional enum.EditPostControlReason BizReason;
	4: optional EditPostVisibilityPermission VisibilityPermission;
	// Descriptions on tree:
	// ContentModel.StandardBiz.EditPost.Permission.BizPermissions.IsRecyclable - if the post is recyclable
	5: optional bool IsRecyclable;
}

struct EditPostPermissionStruct {
	1: optional list<EditPostBizPermissionStruct> BizPermissions;
}

struct EffectCreatorStruct {
	// extra.is_effect_designer==1，数据写在item中的，不用放user。各自保留一份吧
	1: optional bool IsEffectCreator;
	// default null, 1 stands need sign, 0 stands no
	2: optional i32 EffectDesignerStatus;
}

struct EffectStruct {
	1: optional EffectCreatorStruct EffectCreator;
}

struct EmojiRecommend {
	1: optional string Emoji;
	2: optional i32 Score;
}

struct EmojiSliderStickerStruct {
	1: optional string AndroidVer;
	2: optional string IOSVer;
	3: optional string EmojiContent;
	4: optional string PromptText;
	5: optional i32 UserSliderValue;
	6: optional i64 AnswerCount;
	7: optional i64 AnswerAvg;
}

struct FakeLandscapeVideoInfoStruct {
	1: optional double Top;
	2: optional double Bottom;
	3: optional double Left;
	4: optional double Right;
	5: optional i32 FakeLandscapeVideoType;
}

struct FollowUpPublishStruct {
	// the original item id which the follow up publish video is shot from, more info https://bytedance.feishu.cn/docs/doccniUL5i3GwexwuzCiFrfwc0e
	1: optional i64 FollowUpPublishFromId;
	// track the original from item id of a follow up published video
	2: optional i64 FollowUpFirstItemId;
	// track the trace from item id of a follow up published video
	3: optional string FollowUpItemIdGroups;
}

struct ForwardStruct {
	// 转发原片对应的item_id
	1: optional i64 OriginItemId;
	// 当前转发的上一级转发的PreForwardId; 供推荐使用
	2: optional i64 PreForwardItemId;
}

struct GameDetailStruct {
	1: optional enum.GameTypeEnum Type;
	2: optional i64 Score;
}

struct GamePartnership {
	// Descriptions on tree:
	// ContentModel.StandardBiz.GamePartnership.PublisherTaskID - TaskID for this publisher Task
	1: optional i64 PublisherTaskID;
	// Descriptions on tree:
	// ContentModel.StandardBiz.GamePartnership.GameID - GameID of this Game
	2: optional i64 GameID;
	// Descriptions on tree:
	// ContentModel.StandardBiz.GamePartnership.GameTagID - ID of this GameTag
	3: optional i64 GameTagID;
	// Descriptions on tree:
	// ContentModel.StandardBiz.GamePartnership.PostFrom - Where is this item post
	4: optional i32 PostFrom;
}

struct GameStruct {
	// 小游戏结构
	1: optional GameDetailStruct GameDetail;
}

struct GeoFenceStruct {
	1: optional list<string> GeoFencing;
	// The list of regions where the video is distributed.
	2: optional list<string> PersonGeoFencing;
}

struct GreenScreenMaterialStruct {
	1: optional enum.GreenScreenType Type;
	2: optional i64 StartTime;
	3: optional i64 EndTime;
	4: optional string ResourceId;
	5: optional string AuthorName;
	6: optional string EffectId;
}

struct GreenScreenStruct {
	1: optional list<GreenScreenMaterialStruct> GreenScreenMaterials;
}

struct HashTagStickerStruct {
	1: optional string HashTagName;
	2: optional i64 HashTagId;
	// 0: unavailable 1: available
	3: optional i32 Status;
}

struct HybridLabelStruct {
	1: optional string BackgroundColor;
	2: optional string Text;
	3: optional string TextColor;
	4: optional common.UrlStruct Image;
	5: optional string RefUrl;
	6: optional enum.HybridLabelTypeEnum LabelType;
}

struct IMConversationStruct {
	// Descriptions on tree:
	// ContentModel.StandardBiz.IMInfo.ConversationInfo.GroupName - the name of the group
	1: optional string GroupName;
	// Descriptions on tree:
	// ContentModel.StandardBiz.IMInfo.ConversationInfo.GroupAvatarUrl - the url of group icon/avatar/logo
	2: optional string GroupAvatarUrl;
	// Descriptions on tree:
	// ContentModel.StandardBiz.IMInfo.ConversationInfo.ConversationID - id of the conversation
	3: optional string ConversationID;
	// Descriptions on tree:
	// ContentModel.StandardBiz.IMInfo.ConversationInfo.ConversationType - type of the conversation in IM (private / group)
	4: optional enum.IMConversationType ConversationType;
}

struct IMStruct {
	// Descriptions on tree:
	// ContentModel.StandardBiz.IMInfo.RecType - Indicate the reason why this video is recommended to FYP
	1: optional enum.IMRecommendType RecType;
	// Descriptions on tree:
	// ContentModel.StandardBiz.IMInfo.DisplayQuickReplyBox - whether to display the quick reply box
	2: optional bool DisplayQuickReplyBox;
	// Descriptions on tree:
	// ContentModel.StandardBiz.IMInfo.FromUserID - the uid of the original video sharer
	3: optional i64 FromUserID;
	4: optional IMConversationStruct ConversationInfo;
}

struct ImagePostStandardStruct {
	1: optional string PostExtra;
	// Descriptions on tree:
	// ContentModel.StandardBiz.ImagePostInfo.PhotoModeImageQualityInfo - trace all image quality strategies
	2: optional string PhotoModeImageQualityInfo;
}

struct InteractionStickerOtherInfoStruct {
	// identifies if the text stickers are translatable
	1: optional bool IsTextStickerTranslatable;
	// the major language through all text stickers
	2: optional string TextStickerMajorLang;
	// identifies if the burn-in captions are translatable
	3: optional bool IsBurnInCaptionTranslatable;
}

// Short User Info for TAG
struct InteractionTaggedUserStruct {
	1: optional i64 UserId;
	// 理论上改字段应该加到UserGroup里，属于用户与用户的关系
	2: optional enum.InteractionTagInterestLevel InterestLevel;
	// 理论上改字段应该加到UserGroup里，属于用户与用户的关系
	3: optional bool IsBusinessAccount;
	// 理论上改字段应该加到UserGroup里，属于用户与用户的关系
	5: optional enum.TaggingBaInvitationStatus InvitationStatus;
}

struct InteractionTagStruct {
	// 理论上改字段应该加到UserGroup里，属于用户与用户的关系
	1: optional enum.InteractionTagInterestLevel InterestLevel;
	2: optional string VideoLabelText;
	// sorted in tag prd requirement
	3: optional list<i64> TaggedUserIds;
	4: optional list<InteractionTaggedUserStruct> TaggedUsers;
}

struct InteractiveEmojiStickerStruct {
	1: optional string AndroidVer;
	2: optional string IOSVer;
	3: optional string EmojiContent;
}

struct ItemCreateAttributeStruct {
	// how the item is created 0:upload, 1: shoot
	1: optional i32 Original;
	// Whether the post is composed by multiple footage
	2: optional i32 IsMultiContent;
	// The content type of item, can be "photo_canvas"，"multi_photo"，"slideshow"，"now", more specific than aweme_type.
	3: optional string ContentType;
	// The name of tab when you're shooting a video or photo, the value can be "photo"、"story"、"now".
	4: optional string ShootTabName;
}

struct ItemStarAltasLinkStruct {
	1: optional i64 Id;
	2: optional i64 OrderId;
	3: optional string Title;
	4: optional string SubTitle;
	5: optional common.UrlStruct AvatarIcon;
	// H5 landing page
	6: optional string WebUrl;
	7: optional string OpenUrl;
}

struct ItemTalentStruct {
	// New version of talent add product link
	1: optional string GoodsRecUrl;
	// Add product link
	2: optional string ManageGoodsUrl;
	// Star map order ID
	3: optional i64 StarAtlasOrderId;
	// Star map status
	4: optional i32 StarAtlasStatus;
	// Star map link information
	5: optional ItemStarAltasLinkStruct StarAtlasLinkInfo;
	// TCM order status: 1 - closed
	6: optional i32 TcmStatus;
	// Disable visibility operations
	7: optional bool PreventPrivacy;
	// Why visibility operations are prohibited
	8: optional string PreventPrivacyReason;
	// TCM review status
	9: optional enum.TCMReviewStatusEnum TCMReviewStatus;
}

struct LabelDetailStruct {
	1: optional i32 Type;
	2: optional string Text;
	3: optional common.UrlStruct Url;
	4: optional string Color;
	5: optional string ColorText;
	6: optional string RefUrl;
	// starling key for text
	7: optional string TextKey;
	// relation label recommend type, from recommend service
	8: optional i32 RecommendType;
}

struct LiftMarkStruct {
	// extra.lift_mark.key
	1: optional string Key;
	// extra.lift_mark.value
	2: optional string Value;
}

struct LikeStruct {
	1: optional bool HideDiggCount;
}

struct LinkStickerStruct {
	// Descriptions on tree:
	// ContentModel.StandardBiz.Sticker.InteractionStickers.LinkSticker.LinkAddress - link address
	1: optional string LinkAddress;
	// Descriptions on tree:
	// ContentModel.StandardBiz.Sticker.InteractionStickers.LinkSticker.DisplayText - display text
	2: optional string DisplayText;
	// Descriptions on tree:
	// ContentModel.StandardBiz.Sticker.InteractionStickers.LinkSticker.LinkEnabledStatus - enabled status
	3: optional string LinkEnabledStatus;
	// Descriptions on tree:
	// ContentModel.StandardBiz.Sticker.InteractionStickers.LinkSticker.LinkRiskScore - link risk score
	4: optional i32 LinkRiskScore;
}

struct LiveCountdownStickerStruct {
	1: optional string Title;
	2: optional i64 EndTime;
	3: optional i64 SubscribedCount;
	4: optional bool IsSubscribed;
	5: optional string TextTobeSubscribed;
	6: optional string TextAlreadySubscribed;
	7: optional string TextAlreadyExpired;
	// Descriptions on tree:
	// ContentModel.StandardBiz.Sticker.InteractionStickers.LiveCountdownInfo.EventId - event id of countdown sticker
	8: optional i64 EventId;
}

struct LiveRoomStruct {
	// Room ID of submission for live playback, highlight and screen recording
	1: optional i64 RoomId;
}

struct LocationInfoDetail {
	1: optional string Code;
	2: optional double Confidence;
}

struct LocationStruct {
	1: optional LocationInfoDetail L0;
	2: optional LocationInfoDetail L1;
	3: optional LocationInfoDetail L2;
	4: optional LocationInfoDetail L3;
	5: optional string LocateMethod;
	6: optional LocationInfoDetail L0Exp;
	7: optional LocationInfoDetail L1Exp;
	8: optional LocationInfoDetail L2Exp;
	9: optional LocationInfoDetail L3Exp;
	10: optional string LocateMethodExp;
}

struct LongVideoStruct {
	// position info for long split videos
	1: optional string PartN;
	2: optional double TrailerStartTime;
}

struct MarketDropStruct {
	// Descriptions on tree:
	// ContentModel.StandardBiz.MarketDropInfo.MarketSubType - 营销类型
	1: optional i32 MarketSubType;
}

struct MarketplaceStruct {
	// Descriptions on tree:
	// ContentModel.StandardBiz.MarketplaceInfo.ItemMarketplaceStatus - // marketplace status
	1: optional i32 ItemMarketplaceStatus;
	// Descriptions on tree:
	// ContentModel.StandardBiz.MarketplaceInfo.CurrencySymbol - currency symbol
	2: optional string CurrencySymbol;
	// Descriptions on tree:
	// ContentModel.StandardBiz.MarketplaceInfo.Condition - marketplace condition
	3: optional i32 Condition;
	// Descriptions on tree:
	// ContentModel.StandardBiz.MarketplaceInfo.Category - marketplace category
	4: optional i32 Category;
	// Descriptions on tree:
	// ContentModel.StandardBiz.MarketplaceInfo.CurrentPrice - merchandise current price
	5: optional double CurrentPrice;
}

// typedef common.MaskDetailStruct MaskDetailStruct
struct MaskStruct {
	// mask infos
	1: optional list<common.MaskDetailStruct> MaskInfos;
	// report mask info (deprecated), use CustomBiz.Mask
	2: optional common.MaskDetailStruct ReportMask;
}

struct MentionStickerStruct {
	1: optional string UserName;
	2: optional string SecUid;
	3: optional string UserId;
	4: optional string Nickname;
	5: optional common.UrlStruct UserAvatarUrl;
	6: optional enum.MentionStickerScenario Scenario;
}

struct MissionInfoStruct {
	// extra.mission_info.id
	1: optional i64 Id;
	// extra.mission_info.status
	2: optional i32 Status;
	// Descriptions on tree:
	// ContentModel.StandardBiz.Commerce.ContentAds.MissionInfo.MissionType - mission type
	3: optional i32 MissionType;
}

struct ContentADsInfoStruct {
	// json string of common commercial info
	1: optional string CommercialVideoInfo;
	// 商业化运营配置数据
	2: optional list<CommerceConfigStruct> CommerceConfigInfos;
	// extra.lift_mark
	3: optional LiftMarkStruct LiftMark;
	// extra.mission_info
	4: optional MissionInfoStruct MissionInfo;
}

struct MomentsModeInfo {
	// Descriptions on tree:
	// ContentModel.StandardBiz.MomentsModeInfo.IsMoments - whether the video/photo is shot from moments mode 0: false 1: true
	1: optional bool IsMoments;
}

struct MufCommentInfoStruct {
	1: optional i64 CommentId;
	2: optional string NickName;
	3: optional common.UrlStruct AvatarThumbUrl;
	4: optional common.UrlStruct AvatarMediumUrl;
	5: optional string Text;
	6: optional bool HasPhoto;
	7: optional i64 UserID;
	8: optional i32 RelationStatus;
}

struct MusicAvatarStruct {
	// 100*100
	1: optional common.UrlStruct AvatarThumb;
	// 168*168
	2: optional common.UrlStruct AvatarThumbnail;
	// 300*300
	3: optional common.UrlStruct AvatarMedium;
	// 720*720
	4: optional common.UrlStruct AvatarLarge;
	// 1080*1080
	5: optional common.UrlStruct AvatarHd;
}

struct MusicDSPStruct {
	1: optional i64 FullClipId;
	2: optional string FullClipAuthor;
	3: optional string FullClipTitle;
	4: optional i32 CollectStatus;
	5: optional MusicAvatarStruct PerformerDefaultAvatar;
	6: optional i64 MvId;
	7: optional bool IsShowEntrance;
}

struct MusicPromotionItemTagInfoStruct {
	// Descriptions on tree:
	// ContentModel.StandardBiz.MusicPromotionContentInfo.MusicPromotionItemTagInfo.TagType - item上的宣推tag类型
	1: optional enum.MusicPromoTagType TagType;
	// Descriptions on tree:
	// ContentModel.StandardBiz.MusicPromotionContentInfo.MusicPromotionItemTagInfo.Text - item展示的tag 文案
	2: optional string Text;
	// Descriptions on tree:
	// ContentModel.StandardBiz.MusicPromotionContentInfo.MusicPromotionItemTagInfo.ShouldShowMusicTitle - 下发tag时是否展示音乐跑马灯
	3: optional bool ShouldShowMusicTitle;
	// Descriptions on tree:
	// ContentModel.StandardBiz.MusicPromotionContentInfo.MusicPromotionItemTagInfo.SchemaUrl - 当tag可点击跳转时的scheme url
	4: optional string SchemaUrl;
}

struct MusicPromotionContentInfoStruct {
	1: optional MusicPromotionItemTagInfoStruct MusicPromotionItemTagInfo;
}

struct MusicStickerStruct {
	// Descriptions on tree:
	// ContentModel.StandardBiz.Sticker.InteractionStickers.MusicSticker.Title - MusicSticker.Title
	1: optional string Title;
	// Descriptions on tree:
	// ContentModel.StandardBiz.Sticker.InteractionStickers.MusicSticker.ArtistName - MusicSticker.ArtistName
	2: optional string ArtistName;
	// Descriptions on tree:
	// ContentModel.StandardBiz.Sticker.InteractionStickers.MusicSticker.Cover - MusicSticker.Cover
	3: optional common.UrlStruct Cover;
	// Descriptions on tree:
	// ContentModel.StandardBiz.Sticker.InteractionStickers.MusicSticker.IsPgc - MusicSticker.IsPgc
	4: optional bool IsPgc;
	// Descriptions on tree:
	// ContentModel.StandardBiz.Sticker.InteractionStickers.MusicSticker.MusicId - MusicSticker.MusicId
	5: optional string MusicId;
	// Descriptions on tree:
	// ContentModel.StandardBiz.Sticker.InteractionStickers.MusicSticker.StickerStyle - MusicSticker.StickerStyle
	6: optional i32 StickerStyle;
	// Descriptions on tree:
	// ContentModel.StandardBiz.Sticker.InteractionStickers.MusicSticker.MusicStickerFrom - MusicSticker.MusicStickerFrom
	7: optional string MusicStickerFrom;
}

struct MutualStruct {
	1: optional enum.MutualType MutualType;
	2: optional list<i64> UserList;
	3: optional i64 Total;
}

struct MatchedFriendLabelStruct {
	// recommend text
	1: optional string Text;
	2: optional string TextColor;
	3: optional string BackgroundColor;
	// mutual relation info
	4: optional MutualStruct MutualInfo;
	// for relation label bury
	5: optional string SocialInfo;
	6: optional string RecType;
	7: optional string FriendTypeStr;
	8: optional i32 RecommendType;
	9: optional string RelationTextKey;
}

struct NatureClassificationStickerStruct {
	// optional SpeciesId; if species_id does not SpeciesId; set as SpeciesId; validation on consumption side
	1: optional i64 SpeciesId;
	// optional SpeciesName; if species_name does not SpeciesName; set as SpeciesName; validation on consumption side
	2: optional string SpeciesName;
	// optional GenusId; genus_id always GenusId; validation on consumption side
	3: optional i64 GenusId;
	// optional GenusName; genus_name always GenusName; validation on consumption side
	4: optional string GenusName;
	// url for redirecting to search landing page
	5: optional string RedirectionUrl;
}

// nearby tab ref Info; more:  https://bytedance.feishu.cn/wiki/wikcnT2jRZoGYTCH97zVMlRTAye
struct NearbyStruct {
	// map<EventTrack; string>, for event track
	1: optional string EventTrack;
	// nearby tab region name which this item was distributed to
	2: optional string NearbyRegion;
	3: optional i64 NearbyVV;
	// item level checker to if "Local Views" tag could be shown
	4: optional bool LocalViewsItemAuth;
}

struct NowButtonInfo {
	// Localized string.
	1: optional string ButtonLabel;
	// Example: https://go.onelink.me/bIdt/409f077
	2: optional string RedirectUri;
}

struct NowForcedVisibility {
	1: optional enum.ForcedVisibleState State;
	2: optional string Message;
}

struct NowIncompatibilityInfo {
	1: optional i32 Reason;
	// Localized string
	2: optional string Explain;
	3: optional NowButtonInfo ResolutionButton;
}

struct NowInteractionControl {
	1: optional bool DisableLike;
	2: optional bool DisableComment;
	3: optional bool DisableReaction;
	// 模糊态点击评论icon
	4: optional enum.InteractionAction BlurCommentAction;
	// 模糊态点击点赞icon
	5: optional enum.InteractionAction BlurLikeAction;
}

// Now Post的属性特征
struct NowPostAttributes {
	// now media type
	1: optional string NowMediaType;
	// now posts visibility
	2: optional i32 NowStatus;
	// 投稿过期时间
	3: optional i64 ExpiredAt;
	// Now投稿的拍摄类型
	4: optional enum.NowPostCameraType NowPostCameraType;
	// 是否是直接拍摄投稿
	5: optional bool IsOriginal;
	6: optional string CreateTimeInAuthorTimezone;
	// 是否包含聚合的now collection
	7: optional bool HasNowCollection;
	// 用户收到push的时间戳
	8: optional i64 LastPushedAtSec;
	// tiktok now incompatibility info
	9: optional NowIncompatibilityInfo IncompatibilityInfo;
}

// Now Post消费信息
struct NowPostConsumptionInfo {
	1: optional NowForcedVisibility ForcedVisibility;
	// now 已读状态
	2: optional enum.NowViewState ViewState;
	// 是否允许now post的交互，主要包括点赞，评论，举报等
	3: optional NowInteractionControl NowInteractionControl;
	// 模糊态展示样式
	4: optional enum.NowBlurType BlurType;
	// 分发的内容来源，主要是好友，热门以及关注的人
	5: optional enum.NowPostSource NowPostSource;
	6: optional map<string,common.UrlStruct> ShareImage;
}

// Now Post的素材内容
struct NowPostContentInfo {
}

struct OpenPlatformDetailStruct {
	1: optional string Id;
	2: optional string Name;
	3: optional common.UrlStruct Icon;
	4: optional string Link;
	5: optional string RawData;
	6: optional string ClientKey;
	7: optional string ShareId;
	// HAS and SHOW the OpenPlatform anchor name.
	8: optional bool ShowAnchorName;
}

struct OpenPlatformStruct {
	1: optional OpenPlatformDetailStruct OpenPlatformDetail;
}

// MinT Operator boost
struct OperatorBoostStruct {
	1: optional i64 TargetVv;
	2: optional i64 EndTime;
	3: optional string Region;
}

// Share to story: This struct contains infos of the original video
struct OriginalSharedVideoInfoStruct {
	// authorId of the original video
	1: optional string OriginalAuthorId;
	// author name of the original video
	2: optional string OriginalAuthorName;
	// awemeId of the original video
	3: optional string OriginalItemId;
	// encrypted authorId
	4: optional string OriginalSecAuthorId;
	// Descriptions on tree:
	// ContentModel.StandardBiz.Sticker.InteractionStickers.VideoShareInfo.OriginalIndex - original photomode post index
	5: optional i32 OriginalIndex;
}

struct OriginalSoundStruct {
	// Playback address for the audio track
	1: optional common.UrlStruct PlayAddr;
}

struct POIAddressInfo {
	1: optional string CityName;
	2: optional string CityCode;
	3: optional string RegionCode;
	4: optional string Lng;
	5: optional string Lat;
	// Descriptions on tree:
	// ContentModel.StandardBiz.PointOfInterest.PoiDetail.AddressInfo.Geohash - POI Geohash info
	6: optional string Geohash;
	// Descriptions on tree:
	// ContentModel.StandardBiz.PointOfInterest.PoiDetail.AddressInfo.Address - poi address
	7: optional string Address;
	// Descriptions on tree:
	// ContentModel.StandardBiz.PointOfInterest.PoiDetail.AddressInfo.SubdivisionCode - L1 geo name ID
	8: optional string SubdivisionCode;
}

struct POIReviewConfigStruct {
	1: optional bool ShowReviewTab;
}

struct POISubTagStruct {
	1: optional string Name;
	2: optional enum.POIAnchorContentType Type;
	3: optional i32 Priority;
}

struct POIAnchorInfo {
	1: optional i64 AnchorId;
	2: optional string Suffix;
	3: optional list<POISubTagStruct> SubTags;
	4: optional enum.POIContentExpType SubTagExpType;
	5: optional i32 SubTagExpTime;
	6: optional bool HasSubArrow;
	7: optional string TrackInfo;
	8: optional list<enum.POIHideType> HideList;
}

struct POIDetailStruct {
	1: optional string PoiName;
	2: optional string PoiId;
	3: optional string PoiType;
	4: optional string InfoSource;
	5: optional string CollectInfo;
	6: optional bool PoiMapKitCollect;
	7: optional i64 VideoCount;
	8: optional POIAddressInfo AddressInfo;
	9: optional POIAnchorInfo VideoAnchor;
	10: optional POIAnchorInfo CommentAnchor;
	11: optional bool IsClaimed;
	12: optional string TypeLevel;
	13: optional POIReviewConfigStruct PoiReviewConfig;
	// Descriptions on tree:
	// ContentModel.StandardBiz.PointOfInterest.PoiDetail.POIClaimStatus - POI 提报状态
	14: optional enum.PoiClaimStatusType POIClaimStatus;
	15: optional common.UrlStruct Icon;
	16: optional common.UrlStruct Thumbnail;
}

struct PaidContentStruct {
	1: optional i64 PaidCollectionId;
	// Descriptions on tree:
	// ContentModel.StandardBiz.PaidContent.Category - Paid Content Series Category (0 Normal, 1 Mini Drama, etc)
	2: optional i64 Category;
	// Descriptions on tree:
	// ContentModel.StandardBiz.PaidContent.EpisodeNum - This field indicates the Series episode number if valid
	3: optional i64 EpisodeNum;
	// Descriptions on tree:
	// ContentModel.StandardBiz.PaidContent.ShouldShowSeriesPurchaseLabel - Paid Content Should Show Series Purchase Label
	4: optional bool ShouldShowSeriesPurchaseLabel;
	// Descriptions on tree:
	// ContentModel.StandardBiz.PaidContent.IsPaidCollectionIntro - Is Paid Collection Intro / Trailer
	5: optional bool IsPaidCollectionIntro;
	// Descriptions on tree:
	// ContentModel.StandardBiz.PaidContent.ShouldShowPreview - Should Show Preview
	6: optional bool ShouldShowPreview;
}

struct PillarBoxVideoInfoStruct {
	1: optional double Top;
	2: optional double Bottom;
	3: optional double Left;
	4: optional double Right;
	5: optional string Version;
}

struct PlaylistStruct {
	// playlist info of the video
	1: optional CreatorPlaylistStruct CreatorPlaylist;
	// whether the current item is not allowed to be added to playlists
	2: optional bool PlayListBlocked;
}

struct PodcastStruct {
	1: optional bool IsListenable;
	2: optional bool IsPodcast;
	// controls how to show follow button on audio feed
	3: optional enum.PodcastFollowDisplay FollowDisplay;
	4: optional string BackgroundColor;
	5: optional bool IsAudible;
	6: optional bool IsPodcastPreview;
	7: optional i64 FullEpisodeItemId;
	8: optional list<string> FullEpisodeAuthors;
	9: optional common.UrlStruct PodcastEpisodePlayAddr;
	10: optional common.UrlStruct PodcastEpisodeCoverImage;
	11: optional string PodcastEpisodeTitle;
	12: optional bool IsEpisodeBrandedContent;
	13: optional i64 PodcastEpisodeDurationMilliseconds;
	14: optional string PodcastFeedTitle;
	15: optional enum.PodcastTnsStatus PodcastEpisodeTnsStatus;
	16: optional enum.PodcastSharkStatus PodcastEpisodeSharkStatus;
}

struct PoiStickerInfoStruct {
	// poi_id of the sticker
	1: optional string PoiId;
}

struct PointOfInterestStruct {
	1: optional POIDetailStruct PoiDetail;
	// Identifies if this item can be re-tag with poi.
	2: optional enum.PoiReTagType PoiReTagSignal;
	// Show text if this item can be re-tag with poi
	3: optional string PoiReTagText;
}

struct PreSaveInfoStruct {
	1: optional list<DSPEntityInfoStruct> DspEntityInfos;
	// Descriptions on tree:
	// ContentModel.StandardBiz.TT2DSPAlbumInfo.PreSaveInfo.Schema - schema router
	2: optional string Schema;
	// Descriptions on tree:
	// ContentModel.StandardBiz.TT2DSPAlbumInfo.PreSaveInfo.EndTime - pre-save campaign end time
	3: optional i64 EndTime;
}

struct PromoteInfoStruct {
	// item has promote entry or not 1 - show; 2 - gray; 3 - not show
	1: optional i32 HasPromoteEntry;
	// if promote entry is gray, pop a toast. starling key
	2: optional string PromoteToastKey;
	// if promote entry is gray, pop a toast. starling writing
	3: optional string PromoteToast;
	// Descriptions on tree:
	// ContentModel.StandardBiz.Commerce.PromoteInfo.CapcutToggle - whether user select Promote check box on Capcut
	4: optional i64 CapcutToggle;
	// Descriptions on tree:
	// ContentModel.StandardBiz.Commerce.PromoteInfo.PromoteIconText - Promote entrance icon text
	5: optional string PromoteIconText;
}

struct QuestionStickerStruct {
	1: optional i64 QuestionId;
	2: optional i64 UserId;
	3: optional i64 ItemId;
	4: optional string Content;
	5: optional string Username;
	6: optional common.UrlStruct UserAvatar;
	7: optional string SecUid;
	8: optional common.ShareDetailStruct InviteShareInfo;
	// extra info stored as json string
	9: optional string Extra;
	// json str map<category_x, category_name> question, used for event tracking; example: '{category_1:foo,category_2:bar}'
	10: optional string CategoryMeta;
}

struct QuickComment {
	// Descriptions on tree:
	// ContentModel.StandardBiz.Interaction.Comment.CommentConfig.QuickComment.Enabled - add a switch to control quick comemnt
	1: optional bool Enabled;
	// Descriptions on tree:
	// ContentModel.StandardBiz.Interaction.Comment.CommentConfig.QuickComment.RecLevel - quick comment recommend level: https://bytedance.larkoffice.com/wiki/Mx4qwaxOSi087Yk89dIc9xNznhc
	2: optional enum.QuickCommentRecLevelEnum RecLevel;
}

struct QuickMentionUser {
	// Descriptions on tree:
	// ContentModel.StandardBiz.Interaction.Comment.CommentConfig.QuickMention.UID - Quick Mention候选人UID
	1: optional i64 UID;
	// Descriptions on tree:
	// ContentModel.StandardBiz.Interaction.Comment.CommentConfig.QuickMention.NickName - Quick Mention候选人nickname
	2: optional string NickName;
	// Descriptions on tree:
	// ContentModel.StandardBiz.Interaction.Comment.CommentConfig.QuickMention.FollowStatus - Quick Mention候选人与当前用户关注关系
	3: optional enum.FollowStatus FollowStatus;
	// Descriptions on tree:
	// ContentModel.StandardBiz.Interaction.Comment.CommentConfig.QuickMention.SecUID - QuickMention候选人Sec UID
	4: optional string SecUID;
}

struct ReactStruct {
}

struct RealTimeComponentConfigStruct {
	// Descriptions on tree:
	// ContentModel.StandardBiz.RealTimeFcpConfig.RealtimeComponentConfig.GlobalShowLimit - 组件全局规避数量限制
	1: optional i32 GlobalShowLimit;
	// Descriptions on tree:
	// ContentModel.StandardBiz.RealTimeFcpConfig.RealtimeComponentConfig.RerankedComponents - 优先级重排过的组件列表
	2: optional list<string> RerankedComponents;
	// Descriptions on tree:
	// ContentModel.StandardBiz.RealTimeFcpConfig.RealtimeComponentConfig.ShowInfoResetComponents - 展示策略重置过的组件列表
	3: optional list<ComponentShowInfoStruct> ShowInfoResetComponents;
	// Descriptions on tree:
	// ContentModel.StandardBiz.RealTimeFcpConfig.RealtimeComponentConfig.FinalFilterComponents - 最终过滤组件列表
	4: optional list<string> FinalFilterComponents;
	// Descriptions on tree:
	// ContentModel.StandardBiz.RealTimeFcpConfig.RealtimeComponentConfig.DisableComponents - 不同题材下 disable 的组件列表
	5: optional list<string> DisableComponents;
	// Descriptions on tree:
	// ContentModel.StandardBiz.RealTimeFcpConfig.RealtimeComponentConfig.ServerEventTrackingExtra - 组件埋点字段
	6: optional string ServerEventTrackingExtra;
}

struct RealTimeFcpConfigStruct {
	// Descriptions on tree:
	// ContentModel.StandardBiz.RealTimeFcpConfig.RealtimeComponentConfig - 组件默认规则配置
	1: optional RealTimeComponentConfigStruct RealtimeComponentConfig;
}

struct RelatedLiveStruct {
	1: optional string Content;
	2: optional string RelatedTag;
	3: optional string TagName;
}

struct RiskWarningStruct {
	1: optional i32 Type;
	2: optional string Content;
	// 下沉内容
	3: optional bool Sink;
	// 自律委员会投票状态
	4: optional bool Vote;
	// 警示
	5: optional bool Warn;
	// 提示
	6: optional bool Notice;
	// 跳转链接
	7: optional string Url;
}

struct SchedulePostStruct {
	1: optional i64 CreateTime;
	2: optional i64 ScheduleTime;
	// original status chose by user
	3: optional i32 OriUserStatus;
	4: optional bool Finish;
}

struct ServerBaseComponentConfigStruct {
	// Descriptions on tree:
	// ContentModel.StandardBiz.ServerBaseFcpConfig.ServerBaseComponentConfig.ServerBaseComponentConfigHash - 默认配置hash
	1: optional string ServerBaseComponentConfigHash;
	3: optional list<ComponentShowInfoStruct> GlobalRankedComponents;
}

struct ServerBaseFcpConfigStruct {
	// Descriptions on tree:
	// ContentModel.StandardBiz.ServerBaseFcpConfig.ServerBaseComponentConfig - 组件服务端默认规则
	3: optional ServerBaseComponentConfigStruct ServerBaseComponentConfig;
}

struct ShareOperationInfo {
	1: optional string OperationID;
	2: optional string OperationName;
	// Descriptions on tree:
	// ContentModel.StandardBiz.ShareOperationInfo.IncentiveOperationType - for tagging the video which needs to display special format
	3: optional string IncentiveOperationType;
}

struct ShareStruct {
	1: optional bool ShowShareLink;
	2: optional common.ShareDetailStruct ShareDetail;
}

struct ShareToStoryStruct {
	// The visibility of the forwarded story will change with the visibility of the original video,
	// Determines whether the forwarded story is visible or not
	1: optional i32 IsVisible;
	2: optional i64 OriginItemID;
	3: optional i32 ShareStoryPostType;
}

struct ShareToVideoStruct {
	1: optional i32 SharePostType;
}

// share to post related info
struct SharePostStruct {
	// share to video related info
	1: optional ShareToVideoStruct ShareToVideoInfo;
}

struct SingleAnchorStruct {
	1: optional enum.AnchorType Type;
	2: optional string Keyword;
	3: optional string Lang;
	4: optional enum.AnchorState State;
	5: optional string Url;
	6: optional i64 Id;
	7: optional string Extra;
}

struct AnchorStruct {
	1: optional AnchorStrategyStruct AnchorStrategy;
	// anchor list
	2: optional list<AnchorCommonStruct> MultiAnchors;
	3: optional SingleAnchorStruct SingleAnchor;
	4: optional string MTEcomAnchorsExtras;
}

struct SocialInteractionAuxiliaryModelStruct {
	// Descriptions on tree:
	// ContentModel.StandardBiz.SocialInteractionAuxiliaryModel.AuxiliaryModelContent - this is for social blob
	1: optional string AuxiliaryModelContent;
}

struct SparkInfoStruct {
	// Descriptions on tree:
	// ContentModel.StandardBiz.SparkInfoStruct.SparkItemId - The field represents the Spark App posts itemId. If this TikTok Item dosen't synced to Spark, this filed is 0.
	1: optional i64 SparkItemId;
	// Descriptions on tree:
	// ContentModel.StandardBiz.SparkInfoStruct.TiktokItemId - The field represents the TikTok App posts itemId.
	2: optional i64 TiktokItemId;
	// Descriptions on tree:
	// ContentModel.StandardBiz.SparkInfoStruct.NeedSyncedToSpark - The field represents this TikTok post whether need synced to Spark App.
	3: optional bool NeedSyncedToSpark;
	// Descriptions on tree:
	// ContentModel.StandardBiz.SparkInfoStruct.SyncedToSpark - The field represents this TikTok post whether synced to Spark App.
	4: optional bool SyncedToSpark;
	// Descriptions on tree:
	// ContentModel.StandardBiz.SparkInfoStruct.PublishSource - 1. TikTok 2. Sparkling 3. SyncedFromTikTok 4. SyncedFromSparkling 5. DualPublish
	5: optional i32 PublishSource;
	// Descriptions on tree:
	// ContentModel.StandardBiz.SparkInfoStruct.label - user label in feed card
	6: optional list<i32> label;
	// Descriptions on tree:
	// ContentModel.StandardBiz.SparkInfoStruct.IsUnavailableRegionContent - The field represents this TikTok post whether created in unsupported country
	7: optional bool IsUnavailableRegionContent;
}

struct StandardComponentInfo {
	1: optional bool BannerEnabled;
	// Descriptions on tree:
	// ContentModel.StandardBiz.StandardComponentInfo.ComponentEventTrackInfo - 埋点信息
	3: optional list<ComponentEventTrackStruct> ComponentEventTrackInfo;
}

struct StickerCommonStruct {
	1: optional string Id;
	2: optional common.UrlStruct IconUrl;
	3: optional string Link;
	4: optional string Title;
	5: optional i32 Type;
	6: optional string Name;
	7: optional string DesignerId;
	8: optional string DesignerEncryptedId;
	9: optional i64 UserCount;
	10: optional list<string> Tags;
	11: optional string OpenUrl;
}

struct StitchStruct {
	// the original video stitched from
	1: optional i64 OriginalAwemeId;
	// The time when video trim starts
	2: optional i64 TrimStartTime;
	// The time when video trim ends
	3: optional i64 TrimEndTime;
}

struct StoryItemStruct {
	1: optional i64 StoryId;
	2: optional bool Viewed;
	3: optional i64 ExpiredAt;
	4: optional i64 TotalComments;
	5: optional bool IsOfficial;
	6: optional i64 ViewerCount;
	7: optional bool ChatDisabled;
	8: optional enum.StoryType StoryType;
	9: optional i64 StoryStyleVersion;
	// Descriptions on tree:
	// ContentModel.StandardBiz.Story.StoryInfo.IsAvatarTriggeredStory - 是否是转发头像的story
	10: optional bool IsAvatarTriggeredStory;
	// Descriptions on tree:
	// ContentModel.StandardBiz.Story.StoryInfo.EmojiDisabled - story support comment, need to hide emoji list
	11: optional bool EmojiDisabled;
	// Descriptions on tree:
	// ContentModel.StandardBiz.Story.StoryInfo.BirthdayCelebrationDisabled - story birthday celebration disable control field
	12: optional bool BirthdayCelebrationDisabled;
}

struct StoryLiteMetadataStruct {
	1: optional i64 ItemID;
	2: optional i64 ProgressBarNum;
	// Descriptions on tree:
	// ContentModel.StandardBiz.Story.UserStoryInfo.AllStoryLiteMetadata.Viewed - story viewed status
	3: optional bool Viewed;
	// Descriptions on tree:
	// ContentModel.StandardBiz.Story.UserStoryInfo.AllStoryLiteMetadata.ExpireAt - story expire time
	4: optional i64 ExpireAt;
	// Descriptions on tree:
	// ContentModel.StandardBiz.Story.UserStoryInfo.AllStoryLiteMetadata.AwemeType - story aweme type
	5: optional i32 AwemeType;
}

struct StoryNoteGradientImgStruct {
	// Descriptions on tree:
	// ContentModel.StandardBiz.StoryNoteInfo.BackgroundInfo.GradientImgData.Colors - a list of colors for for gradient image
	1: optional list<string> Colors;
	// Descriptions on tree:
	// ContentModel.StandardBiz.StoryNoteInfo.BackgroundInfo.GradientImgData.Locations - a list of locations for gradient image
	2: optional list<double> Locations;
}

struct StoryNoteBackgroundStruct {
	// Descriptions on tree:
	// ContentModel.StandardBiz.StoryNoteInfo.BackgroundInfo.BackgroundType - type of background that user chooses
	1: optional enum.StoryNoteBackgroundType BackgroundType;
	2: optional StoryNoteGradientImgStruct GradientImgData;
	// Descriptions on tree:
	// ContentModel.StandardBiz.StoryNoteInfo.BackgroundInfo.BackgroundImgUrl - background image url
	3: optional string BackgroundImgUrl;
	// Descriptions on tree:
	// ContentModel.StandardBiz.StoryNoteInfo.BackgroundInfo.BackgroundGeckoID - gecko id of story note's background image
	4: optional string BackgroundGeckoID;
}

struct StoryNoteInfoStruct {
	// Descriptions on tree:
	// ContentModel.StandardBiz.StoryNoteInfo.TextContent - text entered by users when posting
	1: optional string TextContent;
	2: optional StoryNoteBackgroundStruct BackgroundInfo;
}

struct StreaksMetaStruct {
	// Descriptions on tree:
	// ContentModel.StandardBiz.Streaks.StreaksMeta.StreaksLevel - The streaks level of streaks post
	1: optional i32 StreaksLevel;
}

struct StreaksStruct {
	// Descriptions on tree:
	// ContentModel.StandardBiz.Streaks.IsStreaksPost - Check if the post's type is streaks
	1: optional bool IsStreaksPost;
	2: optional StreaksMetaStruct StreaksMeta;
}

struct SubOnlyVideoGateStruct {
	// Descriptions on tree:
	// ContentModel.StandardBiz.SubOnlyVideo.SubOnlyVideoGates.IsSOVUrllessEnabled - feature gate for SOV urlless aweme
	1: optional bool IsSOVUrllessEnabled;
}

struct SubOnlyVideoInfo {
	// Descriptions on tree:
	// ContentModel.StandardBiz.SubOnlyVideo.ShouldShowPaywall - add this field for mobile to decide if should show paywall for sub only videos
	1: optional bool ShouldShowPaywall;
	// Descriptions on tree:
	// ContentModel.StandardBiz.SubOnlyVideo.SubOnlyVideoGates - store all gates of sub-only video related features
	2: optional SubOnlyVideoGateStruct SubOnlyVideoGates;
}

struct SuggestPromotionInfo {
	// Descriptions on tree:
	// ContentModel.StandardBiz.Search.SuggestPromotionInfo.PromotionText - 促销文案
	1: optional string PromotionText;
	// Descriptions on tree:
	// ContentModel.StandardBiz.Search.SuggestPromotionInfo.SuggestPromotionType - 促销文案类型枚举值
	2: optional enum.SuggestPromotionType SuggestPromotionType;
}

struct SuggestWordStruct {
	1: optional string Word;
	2: optional string WordId;
	3: optional string Info;
	// Descriptions on tree:
	// ContentModel.StandardBiz.Search.SuggestWordList.SuggestWords.Words.PenetrateInfo - 推荐引擎透传给端上的字段
	4: optional string PenetrateInfo;
	// Descriptions on tree:
	// ContentModel.StandardBiz.Search.SuggestWordList.SuggestWords.Words.WordsType - 推荐词类型
	5: optional string WordsType;
}

struct SuggestWordListStruct {
	// suggest word list
	1: optional list<SuggestWordStruct> Words;
	// the icon url before the list
	2: optional common.UrlStruct IconUrl;
	// the scene where the list appears
	3: optional string Scene;
	// the hint text before the list
	4: optional string HintText;
	// the extra info
	5: optional string ExtraInfo;
	// virtual signal of suggest words for FE
	6: optional string QrecVirtualEnable;
}

struct SuggestWordListDetailStruct {
	// suggest word list
	1: optional list<SuggestWordListStruct> SuggestWords;
}

// commerce
struct TCMInfoStruct {
	// branded content accounts of item.see more: https://bytedance.feishu.cn/wiki/wikcnvIxVuSJUShrqBnEezB2Lbh
	1: optional list<BrandContentAccountStruct> BcAccounts;
	2: optional ItemTalentStruct TalentInfo;
	// branded content hashtag new format test
	3: optional string BcAdLabelText;
	// extra.branded_content_type
	4: optional i64 BrandedContentType;
}

struct CommerceStruct {
	1: optional TCMInfoStruct TcmInfo;
	// 标准广告
	2: optional ADInfoStruct AdInfo;
	3: optional PromoteInfoStruct PromoteInfo;
	// 品牌方广告，UGC->广告
	4: optional ContentADsInfoStruct ContentAds;
	5: optional BizAccountStruct BizAccountInfo;
	// Descriptions on tree:
	// ContentModel.StandardBiz.Commerce.BrandOrganicType - bo type
	6: optional i64 BrandOrganicType;
}

struct TT2DSPAlbumStruct {
	1: optional PreSaveInfoStruct PreSaveInfo;
	2: optional DspAuthToken Token;
	// Descriptions on tree:
	// ContentModel.StandardBiz.TT2DSPAlbumInfo.LinkedPlatform - linked dsp platform
	3: optional enum.DspPlatform LinkedPlatform;
}

struct TT2DspContentInfoStruct {
	// Descriptions on tree:
	// ContentModel.StandardBiz.TT2DspContentInfo.Status - tt2dsp 状态字段
	1: optional enum.TT2DspStatus Status;
	// Descriptions on tree:
	// ContentModel.StandardBiz.TT2DspContentInfo.ButtonType - dsp button type
	2: optional enum.TT2DspButtonType ButtonType;
}

struct TTECSuggestWordStruct {
	1: optional string Word;
	2: optional string WordId;
	3: optional string Info;
}

struct TTECSuggestWordListStruct {
	// suggest word list
	1: optional list<TTECSuggestWordStruct> Words;
	// the icon url before the list
	2: optional common.UrlStruct IconUrl;
	// the scene where the list appears
	3: optional string Scene;
	// the hint text before the list
	4: optional string HintText;
	// the extra info
	5: optional string ExtraInfo;
	// virtual signal of suggest words for FE
	6: optional string QrecVirtualEnable;
	// 跳转
	7: optional enum.RedirectPage RedirectPage;
}

struct ECommerceStruct {
	// TTEC suggest word list
	1: optional list<TTECSuggestWordListStruct> TTECSuggestWords;
}

struct TTMStoreLinkStruct {
	1: optional string Link;
	2: optional string Data;
}

struct TTMLinkStruct {
	1: optional string AppLink;
	2: optional string DeepLink;
	3: optional string DownloadLink;
	4: optional TTMStoreLinkStruct StoreLink;
}

struct TTMBrandStruct {
	1: optional string Entrance;
	2: optional TTMLinkStruct Link;
	3: optional string Title;
	4: optional string Subtitle;
	5: optional string ButtonText;
	6: optional bool Subscribed;
	7: optional enum.TTMUaSwitchStatus UaSwitchStatus;
}

struct TTMInfoStruct {
	1: optional enum.TTMProductType Product;
	2: optional TTMBrandStruct Brand;
	3: optional string VIPVerificationSchema;
	// Descriptions on tree:
	// ContentModel.StandardBiz.MusicProduct.TikTokMusic.NextSchema - schema to show ttm download panel
	4: optional string NextSchema;
}

struct MusicProductStruct {
	1: optional TTMInfoStruct TikTokMusic;
}

struct TTSAudioStruct {
	// Subtitle language code provided by VideoArch, e.g., "jpn-JP"
	1: optional string Lang;
	// Subtitle language ID provided by VideoArch
	2: optional i64 LanguageId;
	// Stringified enum value that represents a dubbing voice
	3: optional string VoiceType;
	// Playback address for the audio track
	4: optional common.UrlStruct PlayAddr;
	// Information for the client to adjust audio volume/equilibrium
	5: optional string VolumeInfo;
	// bit rate for this audio file, used for event tracking
	6: optional i32 BitRate;
	// Dubbing language code to match with CLA client CaptionModel
	7: optional string LanguageCode;
}

struct AudioStruct {
	// CDN URL expiration time in UTC with the smallest unit in second
	1: optional i64 CdnUrlExpired;
	// List of TTS audio track information
	2: optional list<TTSAudioStruct> TtsInfos;
	// list of original sounds information
	3: optional list<OriginalSoundStruct> OriginalSoundInfos;
	// Descriptions on tree:
	// ContentModel.StandardBiz.AudioInfo.UsedFullSong - This field will be used for the full song feature. When a video is created, we want to store whether or not it was created using full song or not
	4: optional bool UsedFullSong;
}

struct TTSVoiceInfoStruct {
	// Descriptions on tree:
	// ContentModel.StandardBiz.TTSVoiceInfo.TTSVoiceAttr - This field is a json-serialized list of struct as string that contains extra attributes for each tts_voice effect used in the item
	1: optional string TTSVoiceAttr;
	// Descriptions on tree:
	// ContentModel.StandardBiz.TTSVoiceInfo.TTSVoiceReuseParams - This field is a json-serialized list of struct as string that contains tts reuse params for this item
	2: optional string TTSVoiceReuseParams;
}

struct TextEditStruct {
	// Descriptions on tree:
	// ContentModel.StandardBiz.EditPost.EditPostInfo.TextEditInfo.Description - to be discussed
	2: optional content_base.ContentDescriptionStruct Description;
	// Descriptions on tree:
	// ContentModel.StandardBiz.EditPost.EditPostInfo.TextEditInfo.IsDescNotChanged - if post's description is submitted for change
	3: optional bool IsDescNotChanged;
}

struct EditPostInfoStruct {
	1: optional TextEditStruct TextEditInfo;
	2: optional CoverEditStruct CoverEditInfo;
}

struct EditPostStruct {
	1: optional EditPostPermissionStruct Permission;
	// music edit status
	2: optional i32 MusicEditStatus;
	3: optional EditPostInfoStruct EditPostInfo;
}

struct TextLabelInsetInfoStruct {
	1: optional i32 Top;
	2: optional i32 Trailing;
	3: optional i32 Bottom;
	4: optional i32 Leading;
}

struct AutoCaptionAppearanceStruct {
	// bg_color
	1: optional string BackgroundColour;
	// bg_corner_radius
	2: optional double BackgroundCornerRadius;
	3: optional i32 TextLabelAlignment;
	4: optional list<i32> TextLabelInsets;
	5: optional i32 CaptionTextSize;
	6: optional string CaptionTextColor;
	7: optional double CaptionTextStrokeWidth;
	8: optional string CaptionTextStrokeColor;
	9: optional bool ShouldShowCaptionIcon;
	10: optional bool ShouldShowInstructionText;
	11: optional i32 InstructionTextSize;
	12: optional string InstructionTextColor;
	13: optional double InstructionTextStrokeWidth;
	14: optional string InstructionTextStrokeColor;
	15: optional i32 ExpansionDirection;
	16: optional TextLabelInsetInfoStruct TextLabelInsetInfo;
	17: optional CaptionControlInfoStruct CaptionControlInfo;
}

struct AutoVideoCaptionStickerStruct {
	1: optional enum.AutoCaptionLocationType Location;
	2: optional list<AutoCaptionDetailStruct> AutoCaptions;
	3: optional AutoCaptionPositionStruct Position;
	4: optional AutoCaptionAppearanceStruct Appearance;
}

struct TextPostStruct {
	// Descriptions on tree:
	// ContentModel.StandardBiz.TextPostInfo.IsConvertedText - denotes if text mode post has been converted to photo mode
	1: optional bool IsConvertedText;
	// Descriptions on tree:
	// ContentModel.StandardBiz.TextPostInfo.TextContent - text mode text content
	2: optional string TextContent;
}

struct TextStickerStruct {
	1: optional i32 TextSize;
	2: optional string TextColor;
	3: optional string BgColor;
	4: optional string TextLanguage;
	5: optional double SourceWidth;
	6: optional double SourceHeight;
	7: optional i32 Alignment;
}

struct TextStyleStruct {
	1: optional string Text;
	2: optional string Color;
	3: optional string ColorText;
	4: optional i32 Type;
}

struct LabelStruct {
	1: optional map<string,LabelDetailStruct> Labels;
	// 推荐Label标识，透传推荐的值给客户端
	2: optional string SortLabel;
	3: optional MatchedFriendLabelStruct MatchedFriendLabel;
	// 以下字段从pb_builder下沉过来
	4: optional common.UrlStruct LabelTop;
	// 发起人标签
	5: optional common.UrlStruct LabelOriginAuthor;
	// 音乐首发标签
	6: optional common.UrlStruct LabelMusicStarter;
	// 隐私作品标签
	7: optional common.UrlStruct LabelPrivate;
	// 标签 大图
	8: optional common.UrlStruct LabelLarge;
	// 标签 小图
	9: optional common.UrlStruct LabelThumb;
	// 好友可见作品标签
	10: optional common.UrlStruct LabelFriend;
	// 用于在列表页显示标签
	11: optional list<TextStyleStruct> LabelTopText;
	// 视频文字和样式
	12: optional list<TextStyleStruct> VideoText;
	// 图片+多语言文字下发
	13: optional list<HybridLabelStruct> HybridLabel;
	// 视频的标签列表, 支持feed页多icon
	14: optional list<AwemeLabelStruct> VideoLabels;
	// 音乐首发提示语
	15: optional string LabelMusicStarterText;
}

// ref: https://bytedance.feishu.cn/docx/SL1dd13xIocttLxgCAccLTUYnJe
struct TextToSpeechStruct {
	// tts voice ids have been applied to the video
	1: optional list<string> TtsVoiceIds;
	// referenced tts voice ids have been used to create the video
	2: optional list<string> ReferenceTtsVoiceIds;
	// part of creator tts feature - determines if watermark should be present when downloading. https://bytedance.us.feishu.cn/docx/doxusQZWrHS3XYuGXd5omLClL0e for more info
	3: optional bool ShouldAddCreatorTtsWatermarkWhenDownloading;
}

struct TrendingBarStruct {
	// left icon url
	1: optional common.UrlStruct IconUrl;
	// display text
	2: optional string Display;
	// right button icon url
	3: optional common.UrlStruct ButtonIconUrl;
	// link
	4: optional string Schema;
	// event id
	5: optional i64 EventKeywordId;
	// event name
	6: optional string EventKeyword;
	// client log param
	7: optional string EventTrackingParam;
}

struct TrendingRecallInfoStruct {
	1: optional enum.TrendingProductRecallType TrendingProductRecallType;
}

struct TrendingStruct {
	// trending bar <-
	1: optional TrendingBarStruct TrendingBar;
	// whether to disable search trending bar on the bottom of the video player, more info https://bytedance.feishu.cn/docs/doccnp0B6yyBeEuPAT4f0d81oEd#
	2: optional bool DisableSearchTrendingBar;
}

struct UpvoteInfo {
	// whether cur user reposted
	1: optional bool UserUpvoted;
	// Connected user reposted.
	2: optional string FriendsRecallInfo;
	// Whether cur user want to repost this aweme.
	3: optional i32 RepostInitiatePredict;
}

struct UpvoteStruct {
	// if client need pull upvote info
	1: optional bool NeedUpvoteInfo;
}

struct UsedFullSongStruct {
	// Descriptions on tree:
	// ContentModel.StandardBiz.UsedFullSongInfo.UsedFullSong - This field will tell if a video uses full song or not, which will impact what song clip we play on the music detail page when a user clicks the sound disk from a video.
	1: optional bool UsedFullSong;
}

struct UserFilterStoryStruct {
	// Descriptions on tree:
	// ContentModel.StandardBiz.Story.UserStoryInfo.UserFilterStoryInfo.UserID - story author user id
	1: optional i64 UserID;
	// Descriptions on tree:
	// ContentModel.StandardBiz.Story.UserStoryInfo.UserFilterStoryInfo.FilterStoryIDs - user filter story ids
	2: optional list<i64> FilterStoryIDs;
}

struct UserNowPackStruct {
	// user now consumer status
	1: optional enum.UserNowStatus UserNowStatus;
	// now avatar badge status
	2: optional enum.NowAvatarBadgeStatus NowAvatarBadgeStatus;
}

struct UserNowPostStruct {
	1: optional list<i64> NowPostIDs;
	2: optional i64 TotalCount;
	3: optional i64 NextCursor;
	4: optional i64 PreCursor;
	5: optional bool HasMoreAfter;
	6: optional bool HasMoreBefore;
	7: optional i64 Postion;
	8: optional enum.NowCollectionType CollectionType;
}

struct NowPostStruct {
	// 字段分类Ref: https://bytedance.feishu.cn/docx/DgqEdOYvnoCUPWxsD2NcMD0rnSb
	1: optional NowPostAttributes Attributes;
	2: optional NowPostContentInfo ContentInfo;
	3: optional NowPostConsumptionInfo ConsumptionInfo;
	4: optional UserNowPostStruct UserNowInfo;
	// user now related info https://bytedance.feishu.cn/docx/doxcnnQIexO7A8NghhR1iVjUvbd
	5: optional UserNowPackStruct UserNowPackInfo;
}

struct UserStoryStruct {
	1: optional i32 Total;
	2: optional i32 Postion;
	3: optional list<StoryItemStruct> StoryItemList;
	4: optional i64 MinCursor;
	5: optional i64 MaxCursor;
	6: optional bool HasMoreAfter;
	7: optional bool HasMoreBefore;
	8: optional bool AllViewed;
	// milliseconds
	9: optional i64 LastStoryCreatedTime;
	// for post style single story
	10: optional bool IsPostStyle;
	// user all story metadata
	12: optional list<StoryLiteMetadataStruct> AllStoryLiteMetadata;
	// is story guide card in feed
	13: optional bool IsStoryGuideCard;
	// Descriptions on tree:
	// ContentModel.StandardBiz.Story.UserStoryInfo.ForbiddenToBeGuideCard - 不允许被客户端转化为story卡片
	14: optional bool ForbiddenToBeGuideCard;
	15: optional UserFilterStoryStruct UserFilterStoryInfo;
}

struct StoryStruct {
	// author story status
	1: optional enum.UserStoryStatus StoryStatus;
	2: optional UserStoryStruct UserStoryInfo;
	3: optional StoryItemStruct StoryInfo;
	// share to story related info
	4: optional ShareToStoryStruct ShareToStoryInfo;
}

struct VCFilterInfoStruct {
	// Descriptions on tree:
	// ContentModel.StandardBiz.VCFilterInfo.VCFilterAttr - This field is a json-serialized list of struct as string that contains extra attributes for each vc_filter effect used in the item
	1: optional string VCFilterAttr;
}

struct VideoMentionStruct {
	// Descriptions on tree:
	// ContentModel.StandardBiz.VideoMentionInfo.IsMentioned - if the viewer is mentioned by this video
	1: optional bool IsMentioned;
}

struct VideoReplyCommentStruct {
	// id of the video being replied to
	1: optional i64 AwemeId;
	// id of the ecomment being replied to
	2: optional i64 CommentId;
	// comment id corresponding to the reply video
	3: optional i64 AliasCommentId;
	// username of the user who created the original comment
	4: optional string UserName;
	// comment message corresponding to comment_id
	5: optional string CommentMsg;
	// id of the user who created the original comment
	6: optional i64 CommentUserId;
	// avatar thumbnail of the user who created the original comment
	7: optional common.UrlStruct UserAvatar;
	// not in use
	8: optional bool IsFavorited;
	// whether or not comment is collected
	9: optional bool CollectStat;
}

// 相关视频的入口文案
struct VisualSearchEntryStruct {
	// 主标题
	1: optional string Title;
	// 副标题
	2: optional string SubTitle;
	// 文本颜色
	3: optional string TextColor;
	// 相关视频入口 icon
	4: optional common.UrlStruct IconUrl;
}

struct SearchDetailStruct {
	1: optional string Sentence;
	2: optional string ChallengeId;
	3: optional i64 HotValue;
	4: optional string SearchWord;
	5: optional i64 Rank;
	// Restricted search in this area
	6: optional list<string> RestrictedRegion;
	// Visual search portal (video on or off)
	7: optional bool HasVisualSearchEntry;
	// 热搜词group_id
	8: optional string GroupId;
	// visual search portal (whether the user opens it or not)
	9: optional bool NeedVisualSearchEntry;
	10: optional i32 Label;
	// Hot bar style
	11: optional i32 PatternType;
	// Entry style of related video in feed
	12: optional VisualSearchEntryStruct VisualSearchEntry;
}

struct SearchStruct {
	1: optional SearchDetailStruct SearchDetail;
	// suggest words list
	2: optional SuggestWordListDetailStruct SuggestWordList;
	3: optional SuggestPromotionInfo SuggestPromotionInfo;
}

// ref: https://bytedance.feishu.cn/docx/SL1dd13xIocttLxgCAccLTUYnJe
struct VoiceChangeFilterStruct {
	// voice change filter ids have been applied to the video
	1: optional list<string> VoiceFilterIds;
	// referenced voice change filiter ids have been used to create the video
	2: optional list<string> ReferenceVoiceFilterIds;
}

struct VoteOptionStruct {
	1: optional i64 Id;
	2: optional i64 Count;
	3: optional string Text;
}

struct VoteStruct {
	1: optional i64 Id;
	2: optional i64 CreateTime;
	3: optional i64 CreatorId;
	4: optional i64 EndTime;
	5: optional list<VoteOptionStruct> OptionInfos;
	6: optional string Question;
	7: optional i64 RefId;
	8: optional i64 RefType;
	9: optional i64 SelectedOptionId;
	10: optional i64 StartTime;
	11: optional i32 Status;
	12: optional i64 UpdateTime;
}

struct InteractionStickerStruct {
	1: optional enum.InteractionTypeEnum Type;
	2: optional i32 Index;
	3: optional string Attr;
	5: optional string TrackInfo;
	6: optional VoteStruct VoteInfo;
	7: optional string TextInfo;
	8: optional MentionStickerStruct MentionInfo;
	9: optional HashTagStickerStruct HashTagInfo;
	10: optional LiveCountdownStickerStruct LiveCountdownInfo;
	// [Content.CustomBiz.Sticker.InteractionSticker.AutoCaptionInfo]
	11: optional AutoVideoCaptionStickerStruct AutoCaptionInfo;
	12: optional DuetWithMeStickerStruct DuetWithMe;
	13: optional QuestionStickerStruct QuestionInfo;
	14: optional TextStickerStruct TextStickerInfo;
	15: optional OriginalSharedVideoInfoStruct VideoShareInfo;
	16: optional PoiStickerInfoStruct PoiInfo;
	17: optional NatureClassificationStickerStruct NatureClassificationInfo;
	18: optional bool IsNonGlobal;
	19: optional i32 MaterialIndex;
	22: optional AddYoursStickerStruct AddYoursSticker;
	23: optional CommentPostStickerStruct CommentPostSticker;
	// Descriptions on tree:
	// ContentModel.StandardBiz.Sticker.InteractionStickers.MusicSticker - tech solution doc: https://bytedance.us.larkoffice.com/docx/ChGGdSoxOoFIQ1xxP0IuizY4sag
	24: optional MusicStickerStruct MusicSticker;
	25: optional InteractiveEmojiStickerStruct InteractiveEmojiSticker;
	26: optional EmojiSliderStickerStruct EmojiSliderSticker;
	27: optional LinkStickerStruct LinkSticker;
	28: optional AttributionLinkStickerStruct AttributionLinkSticker;
}

struct StickerStruct {
	1: optional list<StickerCommonStruct> StickerInfos;
	// 发布视频时使用的贴纸
	2: optional string StickerIds;
	// 信息化贴纸
	3: optional list<InteractionStickerStruct> InteractionStickers;
	4: optional InteractionStickerOtherInfoStruct InteractionStickerOtherInfo;
	// 5: optional StickerCommonStruct JDStickerDetail,
	6: optional StickerCommonStruct SPStickerDetail;
	7: optional StickerCommonStruct NMStickerDetail;
}

struct ZeroCommentButtonConfigStruct {
	// Descriptions on tree:
	// ContentModel.StandardBiz.Interaction.Comment.CommentConfig.ZeroCommentButtonConfig.ZeroCommentButtonMainText - button main text
	2: optional string ZeroCommentButtonMainText;
	// Descriptions on tree:
	// ContentModel.StandardBiz.Interaction.Comment.CommentConfig.ZeroCommentButtonConfig.ZeroCommentButtonText - zero comment button text
	3: optional string ZeroCommentButtonText;
	// Descriptions on tree:
	// ContentModel.StandardBiz.Interaction.Comment.CommentConfig.ZeroCommentButtonConfig.ZeroCommentButtonEnable - enable zero comment button
	4: optional bool ZeroCommentButtonEnable;
	// Descriptions on tree:
	// ContentModel.StandardBiz.Interaction.Comment.CommentConfig.ZeroCommentButtonConfig.ZeroCommentButtonImageURL - zero comment button image url
	5: optional common.UrlStruct ZeroCommentButtonImageURL;
	// Descriptions on tree:
	// ContentModel.StandardBiz.Interaction.Comment.CommentConfig.ZeroCommentButtonConfig.ImageType - image type for client upload event
	6: optional i32 ImageType;
}

struct ZeroCommentOptConfig {
	// Descriptions on tree:
	// ContentModel.StandardBiz.Interaction.Comment.CommentConfig.ZeroCommentOptConfig.ExpireTime - cache time for zero-comment videos forbidden comment request
	1: optional i64 ExpireTime;
}

// ref https://bytedance.feishu.cn/docx/DofkdfKzjopdT3xHmeNcWffAn2e
struct CommentConfigStruct {
	// icon text when comment count is zero
	1: optional string ZeroIconText;
	// input box text when comment count is zero
	2: optional string ZeroInputBoxText;
	// input box text when comment count is zero
	3: optional string NonZeroInputBoxText;
	// box text when comment count is zero
	4: optional string EmptyListText;
	// recommend emoji list
	5: optional list<EmojiRecommend> EmojiRecommendList;
	// Descriptions on tree:
	// ContentModel.StandardBiz.Interaction.Comment.CommentConfig.CommentPreload - reload for comment
	6: optional CommentPreload CommentPreload;
	7: optional QuickComment QuickComment;
	// Descriptions on tree:
	// ContentModel.StandardBiz.Interaction.Comment.CommentConfig.ZeroCommentButtonConfig - zero comment button config
	8: optional ZeroCommentButtonConfigStruct ZeroCommentButtonConfig;
	9: optional QuickMentionUser QuickMention;
	10: optional ZeroCommentOptConfig ZeroCommentOptConfig;
	11: optional list<EmojiRecommend> QuickCommentEmojiRecList;
	12: optional list<EmojiRecommend> LongPressEmojiRecList;
}

struct CommentStruct {
	// item comment config，including comment icon text  ..
	1: optional CommentConfigStruct CommentConfig;
	2: optional list<MufCommentInfoStruct> MufCommentInfo;
	3: optional CommentFilterStrategyStruct FilterStrategy;
	4: optional list<CommentTopBarStruct> CommentTopBars;
}

struct InteractionStruct {
	1: optional CommentStruct Comment;
	2: optional LikeStruct Like;
	3: optional CollectStruct Collect;
	4: optional DownloadStruct Download;
	5: optional ForwardStruct Forward;
	6: optional UpvoteStruct Upvote;
	7: optional ShareStruct Share;
	// 同步到其他平台的信息
	8: optional DescendantStruct Descendant;
	9: optional InteractionTagStruct InteractionTag;
	10: optional DuetStruct Duet;
	11: optional ReactStruct React;
	12: optional StitchStruct Stitch;
	13: optional FollowUpPublishStruct FollowUpPublish;
	14: optional UpvoteInfo UpvoteInfo;
}

// ------ Standard Biz ------
struct ContentStandardBizStruct {
	1: optional MaskStruct Mask;
	2: optional LabelStruct Label;
	3: optional AnchorStruct Anchor;
	4: optional BottomBarStruct BottomBar;
	5: optional RiskWarningStruct RiskWarning;
	6: optional ChallengeStruct Challenge;
	7: optional StickerStruct Sticker;
	8: optional InteractionStruct Interaction;
	9: optional ActivityStruct Activity;
	10: optional LongVideoStruct LongVideo;
	11: optional PlaylistStruct Playlist;
	12: optional StoryStruct Story;
	13: optional TrendingStruct Trending;
	14: optional GameStruct Game;
	15: optional LocationStruct Location;
	16: optional LiveRoomStruct LiveRoom;
	17: optional SearchStruct Search;
	18: optional CommerceStruct Commerce;
	19: optional OpenPlatformStruct OpenPlatform;
	20: optional GeoFenceStruct GeoFence;
	21: optional VideoReplyCommentStruct VideoReplyComment;
	22: optional BodydanceStruct Bodydance;
	23: optional EffectStruct Effect;
	24: optional GreenScreenStruct GreenScreen;
	// pb_builder下沉
	25: optional NearbyStruct Nearby;
	26: optional PointOfInterestStruct PointOfInterest;
	27: optional NowPostStruct NowPost;
	28: optional TextToSpeechStruct TextToSpeech;
	// voice change filter related fields should be wrapped into VoiceChangeFilterStruct
	29: optional VoiceChangeFilterStruct VoiceChangeFilter;
	30: optional AudioStruct AudioInfo;
	31: optional MusicDSPStruct MusicDsp;
	32: optional PaidContentStruct PaidContent;
	33: optional PodcastStruct Podcast;
	34: optional CapCutTemplateStruct CapCutTemplate;
	35: optional MusicProductStruct MusicProduct;
	36: optional EditPostStruct EditPost;
	37: optional AnimatedImageUploadStruct AnimatedImageUpload;
	38: optional SchedulePostStruct SchedulePost;
	39: optional DanmakuStruct Danmaku;
	40: optional ECommerceStruct ECommerce;
	41: optional list<OperatorBoostStruct> OperatorsBoostInfo;
	// some attributes about creating item
	42: optional ItemCreateAttributeStruct ItemCreateAttribute;
	43: optional FakeLandscapeVideoInfoStruct FakeLandscapeVideoInfo;
	44: optional CreatorAnalyticsStruct CreatorAnalytics;
	45: optional BatchPostStruct BatchPost;
	46: optional RelatedLiveStruct RelatedLiveInfo;
	47: optional TrendingRecallInfoStruct TrendingRecallInfo;
	48: optional AddYoursRecommendationStruct AddYoursRecommendationInfo;
	49: optional PillarBoxVideoInfoStruct PillarBoxVideoInfo;
	50: optional AigcInfoStruct AigcInfo;
	51: optional ImagePostStandardStruct ImagePostInfo;
	52: optional list<BannerStruct> Banners;
	53: optional ArtistStruct Artist;
	54: optional CommentPostStruct CommentPostInfo;
	55: optional StandardComponentInfo StandardComponentInfo;
	56: optional ShareOperationInfo ShareOperationInfo;
	57: optional SharePostStruct SharePostInfo;
	// Descriptions on tree:
	// ContentModel.StandardBiz.ClinetAIExtraInfo - for client ai的信息字段
	58: optional ClientAIExtraInfo ClinetAIExtraInfo;
	// this field exists if the viewer is able to add this item to story
	59: optional AddToStoryStruct AddToStoryInfo;
	// Descriptions on tree:
	// ContentModel.StandardBiz.TTSVoiceInfo - This is a map structure that contains info related to tts voice
	60: optional TTSVoiceInfoStruct TTSVoiceInfo;
	// Descriptions on tree:
	// ContentModel.StandardBiz.VCFilterInfo - This is a map structure that contains info related to vc voice
	61: optional VCFilterInfoStruct VCFilterInfo;
	// Descriptions on tree:
	// ContentModel.StandardBiz.Chapter - Store video chapter feature related information
	// https://bytedance.us.larkoffice.com/docx/Q6xAdmkHloZtZkx492LusMPGshe
	62: optional Chapter Chapter;
	// Descriptions on tree:
	// ContentModel.StandardBiz.CoinIncentiveVideoInfo - for tagging a video will be an incentive video or not.
	// incentive video means user will get incentive from sharing this video.
	63: optional CoinIncentiveVideoInfoStruct CoinIncentiveVideoInfo;
	// Descriptions on tree:
	// ContentModel.StandardBiz.CreationInfo - information related to aweme struct creation
	64: optional CreationInfoStruct CreationInfo;
	65: optional MarketDropStruct MarketDropInfo;
	// Descriptions on tree:
	// ContentModel.StandardBiz.TT2DspContentInfo - tt2dsp item相关信息
	66: optional TT2DspContentInfoStruct TT2DspContentInfo;
	// Descriptions on tree:
	// ContentModel.StandardBiz.IMInfo - Business info used in TikTok Messaging domain
	67: optional IMStruct IMInfo;
	68: optional SparkInfoStruct SparkInfoStruct;
	69: optional CollabInfoStruct CollabInfo;
	70: optional SubOnlyVideoInfo SubOnlyVideo;
	71: optional MusicPromotionContentInfoStruct MusicPromotionContentInfo;
	72: optional TextPostStruct TextPostInfo;
	73: optional MomentsModeInfo MomentsModeInfo;
	74: optional MarketplaceStruct MarketplaceInfo;
	75: optional StoryNoteInfoStruct StoryNoteInfo;
	// Descriptions on tree:
	// ContentModel.StandardBiz.RealTimeFcpConfig - fcp实时规则配置
	76: optional RealTimeFcpConfigStruct RealTimeFcpConfig;
	// Descriptions on tree:
	// ContentModel.StandardBiz.ServerBaseFcpConfig - fcp默认规则配置
	77: optional ServerBaseFcpConfigStruct ServerBaseFcpConfig;
	// Descriptions on tree:
	// ContentModel.StandardBiz.C2paInfo - AIGC Provenance Data
	78: optional C2paInfo C2paInfo;
	79: optional UsedFullSongStruct UsedFullSongInfo;
	80: optional AddYoursInfo AddYoursInfo;
	81: optional TT2DSPAlbumStruct TT2DSPAlbumInfo;
	82: optional GamePartnership GamePartnership;
	83: optional VideoMentionStruct VideoMentionInfo;
	84: optional ContentCheckInfo ContentCheckInfo;
	85: optional BaInfo BaInfo;
	86: optional ABRollStruct ABRoll;
	87: optional SocialInteractionAuxiliaryModelStruct SocialInteractionAuxiliaryModel;
	88: optional CelebrationItemStruct CelebrationInfo;
	89: optional StreaksStruct Streaks;
}
