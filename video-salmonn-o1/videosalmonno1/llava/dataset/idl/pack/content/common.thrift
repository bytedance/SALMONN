namespace go tiktok.content.common

include "enum.thrift"

struct InlineLinkStruct {
	1: optional string PlaceHolder;
	2: optional string Text;
	3: optional string Url;
}

struct PreviewStruct {
	// Descriptions on tree:
	// ContentModel.Base.Description.TextExtra.StickerUrl.Preview.Data - 用于图片加载完成前的模糊图占位
	1: optional string Data;
	// Descriptions on tree:
	// ContentModel.Base.Description.TextExtra.StickerUrl.Preview.PreviewType - 用于标记使用哪种算法，客户端使用相应的算法解码模糊图占位
	2: optional i32 PreviewType;
}

struct TextWithInlineLinkStruct {
	1: optional string Text;
	2: optional list<InlineLinkStruct> Links;
}

struct MaskPopTextStruct {
	1: optional TextWithInlineLinkStruct Text;
	2: optional bool IsListItem;
}

struct MaskPopWindowStruct {
	1: optional string PopTitle;
	2: optional list<MaskPopTextStruct> PopContent;
}

struct MaskExtraModuleStruct {
	// 模块类型： 1.pop window
	1: optional i32 ModuleType;
	// 入口按钮文案
	2: optional string BtnText;
	// 跳转链接（如有）
	3: optional string Url;
	// 弹窗结构
	4: optional MaskPopWindowStruct PopWindow;
}

struct MaskDetailStruct {
	// Mask type
	1: optional enum.MaskTypeEnum MaskType;
	// Mask status
	2: optional i32 Status;
	// Mask title
	3: optional string Title;
	// Mask content
	4: optional string Content;
	// cancel Mask label
	5: optional string CancelMaskLabel;
	6: optional MaskExtraModuleStruct PopWindow;
	// 生日编辑入口
	7: optional MaskExtraModuleStruct BirthdayEditModule;
	8: optional MaskExtraModuleStruct PolicyModule;
	9: optional string Extra;
}

struct TimeRangeInDoublePrecision {
	1: optional double Start;
	2: optional double End;
}

struct UrlStruct {
	// 客户端并没有用
	1: optional string Uri;
	// [url1, url2, url3] // 客户端一般使用第一个 url 即可
	2: optional list<string> UrlList;
	3: optional i64 Width;
	4: optional i64 Height;
	// android播放视频时，用来对应视频url
	5: optional string UrlKey;
	// 视频下载大小，单位B
	6: optional i64 DataSize;
	// 视频文件md5
	7: optional string FileHash;
	// 视频文件中前一段的checksum，参考文档：https://bytedance.feishu.cn/space/doc/doccnU2AqfciLKVEzeXMc5qA7Ef
	8: optional string FileCs;
	// 视频解密播放密钥
	9: optional string PlayerAccessKey;
	// 视频唯一文件标识
	10: optional string FileId;
	// Descriptions on tree:
	// ContentModel.Base.Description.TextExtra.StickerUrl.ImageMediaModelPbBase64 - ImageMediaModel pb -> base64 string. Used for Image point to point strategy
	11: optional string ImageMediaModelPbBase64;
	// Descriptions on tree:
	// ContentModel.Base.Description.TextExtra.StickerUrl.ImageMediaModelJson - Image Media Model json, unmarshal to ImageMediaModel in pb_builder
	12: optional string ImageMediaModelJson;
	// Descriptions on tree:
	// ContentModel.Base.Description.TextExtra.StickerUrl.Preview - blurred image as placeholder
	13: optional PreviewStruct Preview;
}

struct BitrateImageInfo {
	1: optional string Name;
	2: optional UrlStruct BitrateImage;
}

struct ImageDetailStruct {
	1: optional UrlStruct DisplayImage;
	2: optional UrlStruct WatermarkImage;
	3: optional UrlStruct NonWatermarkImage;
	4: optional UrlStruct OwnerWatermarkImage;
	5: optional UrlStruct UserWatermarkImage;
	6: optional UrlStruct ThumbnailImage;
	7: optional UrlStruct ThumbnailWithLogoImage;
	8: optional list<BitrateImageInfo> BitrateImages;
	9: optional UrlStruct DynamicImage;
}

struct ShareDetailStruct {
	// share description
	1: optional string Desc;
	// share title
	2: optional string Title;
	// share url
	3: optional string Url;
	// share content desc
	4: optional string ContentDesc;
	// share image url
	5: optional UrlStruct ImageUrl;
	// share link desc
	6: optional string LinkDesc;
	// share personal link qr code
	7: optional UrlStruct QrcodeUrl;
	// 0 valid for 6 months, 1 Long term effectiveness
	8: optional i32 PersistStatus;
	// fb ins share content
	9: optional string Quote;
	// jp share content
	10: optional string SignatureDesc;
	// jp share url
	11: optional string SignatureUrl;
	// whatsapp share content
	12: optional string WhatsappDesc;
}
