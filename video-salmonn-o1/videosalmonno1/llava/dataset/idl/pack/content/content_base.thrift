namespace go tiktok.content.base

include "common.thrift"
include "enum.thrift"
include "video.thrift"

struct ACLCommonStruct {
	// 0 allow， otherwise disallow
	1: optional i32 Code;
	// UI show style 0 none btn，1 gray btn， 2 highlight btn
	2: optional i32 ShowType;
	// toast i18n text
	3: optional string ToastMsg;
	// extra
	4: optional string Extra;
	// 1 none ， 2 client-watermark 3 server-watermark
	5: optional i32 Transcode;
	// true Mute; otherwise don't mute
	6: optional bool Mute;
	// popup msg
	7: optional string PopupMsg;
	// see https://bytedance.feishu.cn/sheets/shtcn2fcRwigPHNqnqqaEPKyJvb
	8: optional string PlatformId;
	// see https://bytedance.feishu.cn/sheets/shtcnDFfhhlEnx8s0bM4F86H3Fg?sheet=075bd2
	9: optional string ActionId;
}

struct ACLStruct {
	1: optional ACLCommonStruct DownloadGeneral;
	2: optional map<string,ACLCommonStruct> DownloadOther;
	3: optional i32 ShareListStatus;
	4: optional ACLCommonStruct ShareGeneral;
	5: optional list<ACLCommonStruct> PlatformList;
	6: optional ACLCommonStruct ShareThirdPlatform;
	// actions in share panel
	7: optional map<string,ACLCommonStruct> ShareActionList;
	// actions in long press panel
	8: optional map<string,ACLCommonStruct> PressActionList;
}

struct AppContextStruct {
	1: optional i32 AppId;
	2: optional i64 DeviceId;
	// x.x.x格式版本号
	3: optional string VersionCode;
	4: optional string PriorityRegion;
	5: optional string StoreRegion;
}

struct ArbiterStatusStruct {
	// extra.last_status
	1: optional i32 LastStatus;
	// 参考 tiktok/item_visibility_mapping
	2: optional i32 UserStatus;
	// 参考 tiktok/item_visibility_mapping
	3: optional i32 ContentReviewStatus;
}

struct ContentAuthorStruct {
	1: optional i64 AuthorUid;
}

struct ContentBasicStruct {
	1: optional AppContextStruct AppContext;
	2: optional i32 ContentType;
	// should be deprecated later
	3: optional i32 MediaType;
	4: optional i64 CreateTime;
	5: optional i64 ModifyTime;
}

struct ContentBizTagsStruct {
	// extra.is_long_video
	1: optional bool IsLongVideo;
	// 通过10个商业化字段计算得来
	2: optional bool IsCommerceVideo;
	3: optional bool IsPgcShow;
	4: optional bool IsSpecialVr;
	// extra.is_paid_content
	5: optional bool IsPaidContent;
	// 是否是推广视频
	6: optional bool IsAds;
	// 是否是舒缓视频
	7: optional bool IsRelieve;
	// 是否是hash tag
	8: optional bool IsHashTag;
	// 是否是调查问卷视频
	9: optional bool WithSurvey;
	// 是否是熟人视频／图片等 is item from familiar
	10: optional bool IsFamiliar;
	11: optional bool IsTop;
	// 是否带有商品 <-
	12: optional bool WithGoods;
	// The special mode of SpecialMode; judgment comes from the item extra is_teen_video field
	13: optional bool AvailableForTeen;
	// if a video is created from past memory entrance, 0:no 1:yes, ref: https://bytedance.feishu.cn/docx/JxNodsNOTok1ZTxKqrjcvHjWncf
	14: optional bool IsOnThisDay;
	// identify current video is sub only video or not
	15: optional bool IsSubOnlyVideo;
	// decide if a post is tiktok story
	16: optional bool IsTikTokStory;
	// decide if a post is switched by story
	17: optional bool IsStoryToNormal;
}

// 当前和VideoCoverStruct内容一致
struct ContentCoverStruct {
	// Default/Origin cover: first not-all-black frame
	1: optional common.ImageDetailStruct OriginalCover;
	// User selected static frame cover。视频封面地址  v1.1.0 之前大图小图用这一个字段, v1.1.0 之后表示小图
	2: optional common.ImageDetailStruct SelectedCover;
	// 动态封面 Frame 9
	3: optional common.ImageDetailStruct AnimatedCoverF9;
	// 动态封面 Frame 6
	4: optional common.ImageDetailStruct AnimatedCoverF6;
	5: optional double SelectedCoverTsp;
}

struct ContentGroupStruct {
	1: optional i64 GroupId;
	2: optional list<i64> GroupIdList0;
	3: optional list<i64> GroupIdList1;
}

struct ContentSoundStruct {
	1: optional i64 SoundId;
	2: optional double Volume;
}

struct ImagePIPStruct {
	1: optional common.ImageDetailStruct Front;
	2: optional common.ImageDetailStruct Back;
	3: optional common.ImageDetailStruct FrontFuzzy;
	4: optional common.ImageDetailStruct BackFuzzy;
	5: optional common.ImageDetailStruct CompositeLtr;
	6: optional common.ImageDetailStruct CompositeRtl;
	7: optional common.ImageDetailStruct CompositeLtrFuzzy;
	8: optional common.ImageDetailStruct CompositeRtlFuzzy;
}

struct ImageSlideStruct {
	1: optional list<common.ImageDetailStruct> Slides;
}

struct InteractPermissionDetailStruct {
	// 0、1、2、3、4 --> enable、disable、hide、disable4all、hide4all
	1: optional enum.InteractPermissionEnum Status;
	2: optional string DisableToast;
	// the interctive text if the interaction is allowed
	3: optional string InteractionText;
}

struct InteractPermissionStruct {
	// 0、1、2、3、4 --> able、disable、hide、disable4all、hide4all
	1: optional InteractPermissionDetailStruct Duet;
	// 0、1、2、3、4 --> able、disable、hide、disable4all、hide4all
	2: optional InteractPermissionDetailStruct Stitch;
	// video privacy setting control. 0,1,2 --> able,disable,hide
	3: optional InteractPermissionDetailStruct DuetPrivacySetting;
	// video privacy setting control. 0,1,2 --> able,disable,hide
	4: optional InteractPermissionDetailStruct StitchPrivacySetting;
	// 0、1、2、3、4 --> able、disable、hide、disable4all、hide4all
	5: optional InteractPermissionDetailStruct Upvote;
	// 0、1 --> able、disable
	6: optional InteractPermissionDetailStruct AddingToStory;
	7: optional InteractPermissionDetailStruct CreateSticker;
	8: optional InteractPermissionDetailStruct SwitchStoryToPost;
	// 0、1 --> able、disable
	9: optional InteractPermissionDetailStruct AllowAddingAsPost;
}

struct InteractionStatusStruct {
	1: optional bool IsCollected;
	2: optional bool IsUserDigged;
}

struct MissingStatsStruct {
	// 0计数信息(不包括评论)获取成功 1获取失败
	1: optional bool MissingCount;
	// 0评论计数信息获取成功 1获取失败
	2: optional bool MissingCommentCount;
}

struct InteractionStatisticsStruct {
	1: optional MissingStatsStruct MissingStats;
	// 评论数
	2: optional i64 CommentCount;
	// 点赞(收藏) 数
	3: optional i64 DiggCount;
	// 下载次数
	4: optional i64 DownloadCount;
	// 播放次数
	5: optional i64 PlayCount;
	// 分享次数
	6: optional i64 ShareCount;
	// 转发次数
	7: optional i64 ForwardCount;
	// whatsapp渠道分享次数
	8: optional i64 WhatsAppShareCount;
	// number of favorites
	9: optional i64 CollectCount;
	// number of reposts
	10: optional i64 RepostCount;
}

struct OriginalSoundStruct {
	// Descriptions on tree:
	// ContentModel.Base.Audio.OriginalSound.SoundVid - original audio vid
	1: optional string SoundVid;
	// Descriptions on tree:
	// ContentModel.Base.Audio.OriginalSound.Volume - original audio volume
	2: optional double Volume;
	// Descriptions on tree:
	// ContentModel.Base.Audio.OriginalSound.PlayAddr - original audio play url
	3: optional common.UrlStruct PlayAddr;
}

struct PermissionControlStruct {
	// can I download it
	1: optional bool AllowDownload;
	// 能否合拍
	2: optional bool AllowDuet;
	// 能否抢镜
	3: optional bool AllowReact;
	// 能否被Stitch
	4: optional bool AllowStitch;
	// 是否允许分享
	5: optional bool AllowShare;
	// 是否允许评论
	6: optional bool AllowComment;
	// 是否允许进入音乐详情页 // if you can use music, go to the music details page
	7: optional bool AllowUseMusic;
	// show reuse original sound entrance in music detail page
	8: optional bool AllowReuseOriginalSound;
	// whether the user can send gift for this video
	9: optional bool AllowGift;
	// control.allow_dynamic_wallpaper
	10: optional i32 AllowDynamicWallpaper;
	// control.allow_adding_to_story
	11: optional i32 AllowAdd2Story;
	// control.download_type
	12: optional i32 DownloadType;
	// 0 not allowed to share 1 share download 2 share QR code
	13: optional i32 ShareType;
}

struct PermissionStatusStruct {
	// extra.item_duet
	1: optional enum.RecreateLimit DuetStatus;
	// extra.item_react
	2: optional enum.RecreateLimit ReactStatus;
	// 视频维度stitch设置 <-
	3: optional enum.RecreateLimit StitchStatus;
	// extra.item_comment 保持使用bool类型，可读性更好
	4: optional i32 CommentStatus;
	5: optional i32 DownloadStatus;
	6: optional i32 FixedDownloadStatus;
	// extra.without_watermark
	7: optional bool WithoutWatermark;
	8: optional bool PreventDownload;
}

struct InteractionPermissionStruct {
	1: optional PermissionControlStruct PermissionControl;
	2: optional PermissionStatusStruct PermissionStatus;
	// 访问控制字段
	3: optional ACLStruct AccessControl;
	// interact permission
	4: optional InteractPermissionStruct InteractPermission;
}

struct ContentInteractionStruct {
	1: optional InteractionPermissionStruct Permission;
	2: optional InteractionStatusStruct Status;
	3: optional InteractionStatisticsStruct Statistics;
}

struct PrivacyMarkStruct {
	1: optional bool RecreatedFromU16;
	2: optional bool VidDataCorrupted;
	// extra.ftc==1
	3: optional bool DeletedForFtc;
	// extra.user_canceled==1
	4: optional bool DeletedForUserCanceled;
	// 是否删除 <-
	5: optional bool IsDelete;
	// True为 隐私，False为公开, 不传默认为空开
	6: optional bool IsPrivate;
	// 视频是否为自见 true自见 <-Visibility
	7: optional bool IsSelfSee;
	// 视频是否为下架
	8: optional bool IsProhibited;
}

struct RateInfoStruct {
	1: optional i32 Rate;
	2: optional i32 SubRate;
	// Descriptions on tree:
	// ContentModel.Base.Status.RateInfo.ContentLevel - 视频内容评级指标
	3: optional i32 ContentLevel;
}

struct ReviewReasonStruct {
	// 审核理由
	1: optional i32 ReasonType;
	// 审核描述
	2: optional string ReasonDesc;
}

struct ReviewResultStruct {
	// 0 normal; 1 audit off the shelf; 2 audit self see
	1: optional i32 Status;
	// Can you tell me
	2: optional bool ShouldTell;
	// Details page H5
	3: optional string DetailUrl;
	// Red button text under video
	4: optional string VideoDetailNoticeBottom;
	// Prompt text in the middle of the video
	5: optional string VideoDetailNotice;
	// Personal page under the cover layer text
	6: optional string CoverNotice;
}

struct ReviewStatusStruct {
	// 预审状态：业务背景：目前需要给核心创作者添加一个单独的预审通道，在该预审流程中会给审核不通过的用户一个不通过理由，提高他们的视频审核通过率。实现方式为在视频发布时给item_info中添加 is_preview 字段作为标记。 字段含义：打上该标记的item将进入预审队列，走不同的审核流程。客户端识别到is_preview=1，在个人主页会显示为“预审未通过”
	1: optional i32 PreReview;
	// extra.review_status.first_review
	2: optional i32 FirstReview;
	// extra.review_status.second_review
	3: optional i32 SecondReview;
	// extra.review_status.third_review
	4: optional i32 ThirdReview;
	// extra.review_status.ops_review
	5: optional i32 OpsReview;
	// 视频是否已经审核 0未审核 1已审核
	6: optional bool Reviewed;
	// extra.hot_reviewed
	7: optional bool HotReviewed;
	// extra.in_audit_time
	8: optional i64 InAuditTime;
	// 是否在审核中
	9: optional bool InReviewing;
	// 视频下架理由
	10: optional ReviewReasonStruct TakeDownInfo;
	// Video audit status notification
	11: optional ReviewResultStruct ReviewResult;
}

struct SoundMuteStruct {
	1: optional bool IsMute;
	2: optional string MuteDesc;
	// notice tag url
	3: optional string MuteDetailUrl;
	// notice tag text
	4: optional string MuteDetailNoticeBottom;
	// whether sound of video is exempted, Need sound_exempt to bypass music copyright detection for those users who have permissions. Otherwise there videos are still going to get muted
	5: optional enum.SoundExemptionStatus ExemptionStatus;
	6: optional bool IsCopyrightViolation;
	7: optional enum.AudioChangeStatusEnum AudioChangeStatus;
}

struct SoundTrimStruct {
	// video's music begin time
	1: optional i64 BeginTimeInMs;
	// video's music end time
	2: optional i64 EndTimeInMs;
}

struct AddedSoundStruct {
	1: optional i64 SoundId;
	2: optional double Volume;
	3: optional SoundTrimStruct Trim;
	// source from music, from ies.item.info
	4: optional string MusicSelectedFrom;
	// 0: default disable, but if commercial music, need client check; 1: hide title, ref: https://bytedance.feishu.cn/wiki/wikcneRWWSfKvvVkpk4LlSnGkfe
	5: optional i32 MusicTitleStyle;
	// Volume for ImagePost, should be deprecated later.
	6: optional double ImagePostVolume;
}

struct ContentAudioStruct {
	// Content的音频信息，用户原声+配乐 or 用户原声（拆分原声及配乐仍在试验中: https://bytedance.feishu.cn/docx/Jq7LdnkNgoTT6mxi0uUck7TCnwe）
	1: optional ContentSoundStruct Sound;
	// 配乐
	2: optional AddedSoundStruct AddedSound;
	// video mute
	3: optional SoundMuteStruct SoundMute;
	// Descriptions on tree:
	// ContentModel.Base.Audio.OriginalSound - https://libra-sg.bytedance.net/libra/flight/70531236/edit
	4: optional OriginalSoundStruct OriginalSound;
}

struct TextExtraStruct {
	1: optional enum.TextTypeEnum Type;
	2: optional i32 SubType;
	// index for caption lines
	3: optional i64 LineIdx;
	4: optional i64 Start;
	5: optional i64 End;
	6: optional string UserId;
	7: optional string SecretUserId;
	8: optional string AwemeId;
	9: optional i64 HashtagId;
	10: optional string HashtagName;
	11: optional bool IsCommerceHashtag;
	// image StickerId
	12: optional i64 StickerId;
	// image sticker url
	13: optional common.UrlStruct StickerUrl;
	// image sticker source
	14: optional i32 StickerSource;
	// forum question id
	15: optional i64 QuestionId;
	// the id of text extra, used in markup.
	16: optional string TagId;
}

struct ContentDescriptionStruct {
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

struct VideoSingleStruct {
	1: optional string VideoId;
	// 下发类别
	2: optional enum.VideoDistributeTypeEnum DistributeType;
	3: optional video.VideoStruct Video;
}

// 普通视频：1Video+1BGM
// 图文投稿：nImage+1BGM
// NowPostImage：2Image+1BGM
// NowPostVideo Old：1Video+1BGM
// NowPostVideo New：2Video+1BGM, 目前还没有这种形式
struct ContentDisplayStruct {
	1: optional enum.DisplayLayoutEnum Layout;
	2: optional VideoSingleStruct VideoSingle;
	3: optional ImageSlideStruct ImageSlide;
	// picture-in-picture
	4: optional ImagePIPStruct ImagePip;
}

struct VisibilityStruct {
	1: optional enum.PermissionLevelEnum Actual;
	2: optional enum.PermissionLevelEnum UserSet;
	3: optional enum.PermissionLevelEnum ArbiterSet;
	// Descriptions on tree:
	// ContentModel.Base.Status.Visibility.BottomLine - 预设字段，Pack中目前未使用。
	4: optional enum.PermissionLevelEnum BottomLine;
	5: optional i32 CustomVisibilityType;
}

struct ContentStatusStruct {
	// 可见性
	1: optional VisibilityStruct Visibility;
	2: optional RateInfoStruct RateInfo;
	3: optional PrivacyMarkStruct PrivacyMark;
	4: optional ReviewStatusStruct ReviewStatus;
	5: optional ArbiterStatusStruct ArbiterStatus;
	// Descriptions on tree:
	// ContentModel.Base.Status.ItemStatus - status field stored in item-service. NOT recommend to use it directly.
	// https://bytedance.larkoffice.com/wiki/wikcnBQ3qBKhGp3ki9M3uJpKaeh
	6: optional i32 ItemStatus;
}

struct ContentBaseStruct {
	1: optional ContentBasicStruct Basic;
	2: optional ContentGroupStruct Group;
	3: optional ContentStatusStruct Status;
	// Descriptions on tree:
	// ContentModel.Base.BizTags - 该字段仅用于标识视频自身的类型/体裁属性，是ItemType/AwemeType/ContentType的补充。
	// 如果希望增加其他的标志信息，比如“某个视频曾经是否被xxx过”这些信息需要加到StandardBiz下对应的业务结构体中。
	4: optional ContentBizTagsStruct BizTags;
	5: optional ContentAuthorStruct Author;
	// basic information for interactions
	6: optional ContentInteractionStruct Interaction;
	7: optional ContentDescriptionStruct Description;
	8: optional ContentAudioStruct Audio;
	9: optional ContentCoverStruct Cover;
	10: optional ContentDisplayStruct Display;
}
