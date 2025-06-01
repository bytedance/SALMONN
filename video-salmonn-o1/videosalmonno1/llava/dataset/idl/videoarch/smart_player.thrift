include "../base.thrift"
include "../pack/type/enum.thrift"
include "../pack/type/common.thrift"

namespace go toutiao.videoarch.smart_player
namespace py toutiao.videoarch.smart_player

struct VodLibraConfig {
    1: optional string UidABConfig    //uid实验配置
    2: optional string DidABConfig    //did实验配置
}

enum VideoStatus {
    Visibility_L0 = 0 #公开可见，公开视频，所有人可见。具体见状态描述
    Visibility_L1 = 1 #作者/审核可见，只有视频作者/审核人员/审核模型可见，具体见状态描述
    Visibility_L2 = 2 #审核自见，审核人员/审核模型可见，具体见状态描述
    Visibility_L3 = 3 #初始状态，可不用关注此状态, 任何人不可见，包括审核人员和作者自己，具体见状态描述
    Visibility_L4 = 4 #存档状态，只保留原片和一路低清晰度转码流，如果想彻底删除存档视频，见资源清理接入文档
}
enum AuditUrlType  {
    AuditUrlType_InnerOuter = 0 #内外网地址
    AuditUrlType_Inner = 1 #内网地址
    AuditUrlType_Outer = 2 #外网地址
}

enum BigThumbVersion {
    V1 = 0  #单张雪碧图
    V2 = 1  #多张雪碧图
    V3 = 2
    V4 = 3
}

enum VideoDefinition {
    ALL = 0
    V360P = 1
    V480P = 2
    V720P = 3
    V1080P = 4
    V240P  = 5
    V540P  = 6
    HDR    = 7
    V420P  = 8
    V2K =   9
    V4K =   10
}

enum UserPlatform {
    UNKNOWN = 0
    IOS = 1
    ANDROID = 2
    WEB = 3
}

enum DeviceNetwork {
    UNKNOWN = 0
    WIFI = 1
    MOBILE = 2
    NT3G = 3
    NT4G = 4
    NT5G = 5
    WEB = 6
}

#具体使用哪个内网域名，需要预估自己的下载量，使用场景的等。具体联系@gaohonglei @xiangchao
enum VideoPlaySource {
   Extranet = 0    #外网cdn地址
   Intranet = 1    #内网 非重要下载
   Download = 2    #火山lab 视频下载, 非常重要的下载
   VideoAudit = 3  #审核视频
   IntranetAudit = 4 #用于审核组新的审核流程,其他人慎用
   TerminatorDownload = 5 #用于内网下载，走terminator本机房下载
   VL0 = 6   ##外网cdn地地址，和Extranet=0一样
   VL1 = 7   #外网源站地址，通常用于隐私点播
   VL2 = 8   #内网审核源站地址，内网全员可见
   VL3 = 9   #下载地址，内网研发网络可见
   VL4 = 10  #内网办公网地址
}

enum UrlType {
   VL0 = 6   #外网cdn地址
   VL1 = 7   #外网审核源站地址，外网可见
   VL2 = 8   #内网审核源站地址，内网全员可见
   VL3 = 9   #下载地址，内网研发网络可见
   VL4 = 10  #内网办公网地址
   VL5 = 11  # auto 外网地址
}

enum VideoCodecType {
   H264 = 0       #视频h264
   MByteVC1 = 1   #视频bytevc1老格式, (h264+1)
   OByteVC1 = 2   #(h264+1)_hvc1
   Audio_OPUS = 4 #音频opus格式
   Audio_AAC = 5  #音频aac格式
   Audio_MP3 = 6  #音频mp3格式
   ALL = 3        #视频h264和bytevc1等都返回，音频所有格式都返回，通过response中CodecType来区分具体的格式
   ByteVC1 = 7    #视频bytevc1
   AllWithByteVC1 = 8   #视频h264和bytevc1
   ByteVC2 = 9          #视频bytevc2
   AV1 = 10             #AV1
   AllWithByteVC2 = 11  #视频h264/bytevc1/bytevc2
   VP9 = 12       #VP9
   AllWithOByteVC1AndByteVC2 = 13 #视频h264+1/h264+1_hvc1
   Audio_FLAC = 14 #音频flac格式
   Audio_Vorbis = 15 #音频Vorbis格式
}

enum VideoHDRType {
    Normal = 0
    HDR = 1
    HDR10 =2
}

enum VideoCdnType {
   Normal = 0     #普通商业cdn
   P2P = 1        #p2p
   OwnVDP = 2     #自建p2p
   KsyP2P = 3     #金山云p2p
   YunFanP2P = 4  #云帆p2p
   AliyunP2P = 5  #阿里云p2p
   WangsuP2P = 6  #网宿p2p
   OthersP2P = 20 #其他
}

enum ProjectionModelType {
   Normal = 0     #普通视频 Ordinary plane video
   VRERP = 1      #VR视频  普通ERP视频
   VRCubeMap  = 2   #VR视频 普通cubemap视频
   VROffsetCubic =3  #VR视频 offset-cubic视频
   VREac = 4  #VR视频 eac 视频
   VREacOffsetCubic = 5  #VR视频 eac offset-cubic视频
}

#返回不同码率的
enum VideoQualityType {
   Normal  =  0     #正常
   Low     =  1     #低码率
   High    =  2     #高码率
   Medium  =  3
   Lower   =  4
   Lowest  =  5
   Higher  =  6
   Highest =  7
   Veryhigh =  9
   Superhigh = 10
   All     =  8
   Adapt   = 11
   AdaptLow   = 12
   AdaptLower   = 13
   AdaptLowest   = 14
   VLadder = 15
   AdaptHigh = 16
   AdaptHigher = 17
}

enum VideoFormatType {
    Normal = 0 #默认值，音频返回默认m4a,视频返回默认mp4
    MP4    = 1 #视频mp4封装格式
    Audio_M4A = 2 #音频m4a封装格式
    MPD    = 3 #废弃
    M3U8   = 4
    GIF    = 5
    DASH   = 6
    Audio_OGG = 7
    FMP4   = 8
    HLS    = 9
    WEBM    = 10
    Audio_FLAC = 11 #音频flac封装格式
}

enum VideoEncryptionMethod{
    Normal = 0 # 默认值，不加密
    CENC_AES_CTR = 1
    AES_128 = 2
    SAMPLE_AES = 3
}

//CreatorInfo 创作者信息
struct CreatorInfo {
    1: string UserName
    2: i64 CreatorID
}

//签名信息
struct SignInfo {
    1: string AccessKey //vcloud密钥管理系统ak sk
    2: string Date      //%Year%Month%Day 20190110
    3: string Sign      //签名串 hmac_sha256(signed,psm)
}

// PlayTimeLimit
struct PlayTimeLimit {
    1: i64 Start // 开始时间，单位秒 0, 为不限制开始时间
    2: i64 End   // 结束时间，单位秒 0, 为不限制结束时间
}

enum VideoModelVersion {
    V1 = 1
    V2 = 2 #精简版本json
    V3 = 3 #精简版本pb
}

//用于免流判断
enum IsOrderFlow {
    NotOrderFlow = -1
    Unknown = 0
    YesOrderFlow = 1
}

struct Identity {
    1: optional string IdentityInfo             #身份认证信息，以SDK形式提供
    2: optional string AuthToken
    3: optional string AuthPolicy
}

enum EnableDeviceAdaptive {
    Disable = 0                     #设备自适应开关，关闭
    EnableWithDevice = 1            #仅考虑机型等级开启
    EnableWithDeviceAndNet = 2      #综合考虑机型/网络开启
}

enum DeviceBit {
    BitUnknown = 0
    Bit32 = 1
    Bit64 = 2
}

struct UserExtra {
    1: optional string AttrName = "",
    2: optional string AttrValue = "",
}

struct DeviceAdaptive {
    1: optional EnableDeviceAdaptive EnableDeviceAdaptive = EnableDeviceAdaptive.Disable  #设备自适应开关，默认关闭
    2: optional double DeviceScore = 0                                   #机型分数，0-10分，默认 0 获取不到机型分数
    3: optional i64 NetEnergy = 5                                        #(0, 3] --> 弱网，(3, 6] --> 一般网，(6, 7] -->  好网，(7, 8] -->  非常好网，默认5一般网络
    4: optional i64 ScreenWidth = 0                                      #设备屏幕宽
    5: optional i64 ScreenHeight = 0                                     #设备屏幕高
    6: optional i64 NetworkScore = -1                                    #网络质量分数
    7: optional DeviceBit DeviceBit = DeviceBit.BitUnknown               #端上操作系统位数
    8: optional string HostAbi = ""                                      #HostAbi
    9: optional DeviceBit CpuSupportBit = DeviceBit.BitUnknown                          #cpu是支持位数
}

struct ExtraParam {
    1: optional string Caller
    2: optional i64 FromUserId
    3: optional enum.PackLevelEnum PackLevel
    4: optional enum.PackSourceEnum PackSource
    5: optional bool IsDirectPlay
    6: optional bool ProtectCdn
    7: optional bool UseHttps
}

struct UserInfo {
    1: optional i64 DeviceID = 0                                        #如果有，一定要传过来
    2: optional i64 UserID = 0                                          #如果有，一定要带过来
    3: optional i64 WebID = 0
    4: optional UserPlatform Platform = UserPlatform.UNKNOWN            #如果有，一定要传过来
    5: optional i64 Version = 0                                         #int类型version_code版本号，13/32系传递iOS(1.1.1)android(111)统一后的5位int型，IES系透传3位int型版本号
    6: optional i64 AppID = 0                                           #根据自己的业务线填写，一定要带过来
    7: optional string UserIP = ""
    8: optional string OsVersion = ""                                   #如果有，一定要带过来
    9: optional DeviceNetwork Network = DeviceNetwork.UNKNOWN           #如果有，一定要带过来
    10: optional string DeviceType = ""                                 #如果有，一定要带过来
    11: optional string WifiIdentify = ""
    12: optional string PlayerVersion = ""
    13: optional list<string> ABversions                                #ab实验客户端带下来的abversions,方便在server端做ab实验
    14: optional i64 UpdateVersionCode = 0                              #废弃
    15: optional string OpenapiAction = ""                              #网关action字段, smart_player用来做播控策略
    16: optional string ApiUserName = ""                                #video/play接口中user,smart_player用来做播控策略
    17: optional string TopAccountID = ""                               #TopAccountID
    18: optional string VersionCode = ""                                #13/32系系iOS(1.1.1)和android(111)系版本有差异，只需要13/32系app透传即可
    19: optional string CountryCode = "unknown"                         #US IN等
    20: optional i64 ChannelID = 0                                      #频道ID
    21: optional string Channel = ""                                    #渠道信息，对外合作, 配合appid一块使用，
    22: optional i64 IsOrderFlow = IsOrderFlow.Unknown                  #免流判断参数，Unknown是未知，YesOrderFlow 为免流，NotOrderFlow为非免流
    23: optional string Province = ""                                   #省份
    24: optional DeviceAdaptive DeviceAdaptive                          #设备信息
    25: optional i64 UserRole = 0                                       #字幕分发，用户角色 0（未知）1（作者）
    26: optional i64 TtPlayerPluginVersion = 0                          #ttplayer插件版本
    27: optional list<UserExtra> UserExtras                             #自定义用户属性
    28: optional i64 FirstInstallTime = 0                               #最早的一次安装时间
    29: optional string TtPlayerSdkOptions = ""                         #点播sdk配置参数
    30: optional string AbParam,
    31: optional ExtraParam ExtraParam,
    32: optional string PriorityRegion,
    33: optional string MccMnc,
    34: optional i32 isColdStartFeed,
    35: optional i32 FeedPullType,
}

struct UserParams {
    1: required i64 AppID # aid
    2: optional i64 AppVersion #
    3: optional string VersionCode, # 版本号, eg "5.0.1"
    4: optional string PlayerSdkVersion # 点播sdk的版本，eg "1.0.3"
    5: optional i64 DeviceID = 0
    6: optional UserPlatform Platform = UserPlatform.UNKNOWN #iOS andorid  web
    7: optional DeviceNetwork Network = DeviceNetwork.UNKNOWN #3g 4g 5g wifi
    8: optional string DeviceType = "" #设备型号 xiami等
    9: optional string WifiIdentify = "" # wifi名字
}

struct FilterParams {
    1: optional VideoDefinition  NeedDefinition = VideoDefinition.ALL   # 指定分辨率(360p,480p,720p),若没有相应的分辨率,返回其它分辨率
    2: optional VideoCodecType CodecType = VideoCodecType.H264          # 编码格式bytevc1 h264,默认是h264, 若没有bytevc1，则返回h264
    3: optional string Watermark = "default"                            # 水印名称,不明确指定,按照我们服务播放策略返回(若无策略则返回无水印视频)
                                                                        # 如果明确想获取某种水印，需要明确指定。
                                                                        # 如果明确想要获取无水印视频，明确传递(unwatermarked)
    4: optional string StreamType = "normal"                            # 视频转码类型有一下几种，按需传参
                                                                        #normal   普通非加密转码视频流
                                                                        #audio    普通非加密转码音频流
                                                                        #encrypt  加密转码视频流
                                                                        #audio_encrypt    加密转码音频流
                                                                        #normal_short     截断非加密转码视频
                                                                        #perview  预览
    5: optional VideoFormatType FormatType = VideoFormatType.Normal     # 封装格式
    6: optional VideoQualityType VQuality= VideoQualityType.Normal      # 码率质量类型
    7: optional string EncodeUserTag                                    #转码时用户自定义的标签
    8: optional string VLadder = ""               #视频质量业务自定义档位
    9: optional VideoEncryptionMethod EncryptionMethod = VideoEncryptionMethod.Normal  #加密模式
    10: optional ProjectionModelType ProjectionModelType = ProjectionModelType.Normal           #视频投影模式
    11: optional string ForceCodec = ""    #避免设备codec自适应
    12: optional VideoDefinition  MaxDefinition = VideoDefinition.ALL #最高分辨率，NeedDefinition为ALL时生效，下发<=MaxDefinition的多个分辨率
    13: optional list<ProjectionModelType> ProjectionModelTypeList  #指定多个ProjectionModelType，如果非空，将比ProjectionModelType参数更高优
}

struct UrlParams {
    1: optional bool SSL = false
    2: optional UrlType UrlType = UrlType.VL0  # 视频地址类型
    3: optional VideoCdnType CdnType = VideoCdnType.Normal  # 端上支持的CDN类型
    4: optional i64 Indate  # 播放地址的有效期,单位s，联系gaohonglei添加caller白名单。
    5: optional list<VideoCdnType> CDNTypes # CDNType p2p 调度列表,如果非空，将优先使用CDNTypes调度，忽略CdnType
    6: optional PlayTimeLimit PlayTimeLimit  #播放起止时间
    7: optional bool ImmunePolicy = false  #是否豁免policy访问
}

enum Popularity {
    Default = 0
    Cold = -1
    Hot = 1
    ExtremeCold = 2
    Tepid = 3
}

struct BizScheduleScene {
    1: optional Popularity CDNVideoPopularity = Popularity.Default
    2: optional Popularity PCDNVideoPopularity = Popularity.Default
    3: optional Popularity CDNHashVideoPopularity = Popularity.Default
}

struct SeekOffset{
   1: double Opening
   2: double Ending
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

struct Meta {
    1: i64 Height,                   #视频长度
    2: i64  Width,                   #视频宽度
    3: string Format,                #格式(mp4)
    4: double Duration,              #视频长度
    5: i64 Size,                     #视频大小
    6: i64 Bitrate,                  #视频比特率
    7: string Definition             #分辨率
    8: string LogoType               #logo名称
    9: string CodecType              #编码格式
    10: string EncodedType           #视频类型
    11: string TTcopyright           #加密格式
    12: string EncryptionKey         #版权加密钥
    13: string VideoQuality          #高低码率
    14: string FileHash              #md5
    15: string PktOffset             #预加载偏移量
    16: i64 FPS                      #帧率
    17: string EncryptionKeyID       #秘钥keyID
    18: string EncodeUserTag         #转码时指定的usertag
    19: string VLadder               #视频质量业务自定义档位
    20: string QualityDesc           #清晰度描述信息
    21: string EncryptionMethod
    22: string HDRType               #hdr类型
    23: string HDRBit                #hdrBit, 10bit, 8bit...
    24: i64 RealBitRate              #真实比特率
    25: optional string AudioChannels#音频通道
    26: optional string AudioLayout  #音频布局
    27: string PktOffsetMap          #预加载偏移量map
    28: i64 AvgBitrate               #平均比特率
    29: string Mvmaf                 #视频质量评估
    30: double Vqscore               #视频画质评估
    31: string TranscodeFeatureID    #转码策略标识
}

struct PreloadInfo {
   1: i64 SocketBuffer
   2: i64 PreloadSize
   3: i64 PreloadInterval
   4: i64 PreloadMinStep
   5: i64 PreloadMaxStep
   6: i64 UseVideoProxy
}

struct MpsOnlineTranscode {
    1: optional bool SupportOnlineEncrypt  #实时加密
    2: optional bool SupportOnlineWatermark #实时水印
    3: optional map<string, string> MpsSignParams #mps地址签名信息
}

struct FitterInfo{
   1: list<double> FuncParams
   2: double Duration
   3: i64 HeaderSize
   4: i64 FuncMethod
}

struct VolumeInfo {
    1: double Loudness
    2: double Peak
    3: double MaximumMomentaryLoudness
    4: double MaximumShortTermLoudness
    5: double LoudnessRangeStart
    6: double LoudnessRangeEnd
    7: double LoudnessRange
    8: double Version
    9: string VolumeInfoJson
}

struct SandwichInfo {
    1: double Top
    2: double Bottom
    3: double Left
    4: double Right
}

enum SubtitleFileFormat {
    All    = 0
    WebVTT = 1
}

struct SubtitleInfo {                       # 字幕元信息
   1: optional string LanguageID = ""       # 语言标识码ID
   2: optional string LanguageCodeName = "" # 语言标识码
   3: optional string Url = ""              # 字幕地址
   4: optional i64 UrlExpire                # 字幕地址过期时间
   5: optional string Format = ""           # 字幕格式
   6: optional string Version = ""          # 字幕版本
   7: optional string Source = ""           # 字幕来源
   8: optional i32 VideoSubtitleID          # 字幕标识码ID
   9: optional i64 Size                     # 字幕文件大小
   10: optional list<string> Urls           # 字幕主备以及自刷新地址合集
}

struct SubtitleParams {                                               # 字幕请求参数
    1: optional string LanguageCodeName = ""                          # 字幕标识码
    2: optional SubtitleFileFormat Format = SubtitleFileFormat.WebVTT # 字幕格式，默认为WebVTT格式
    3: optional list<string> Versions                                 # 字幕版本列表
}

struct DubParams {                           # 配音请求参数（视频+音频混合流）
    1: optional string LanguageCodeName = "" # 配音字幕标识码
    2: optional list<string> Versions        # 配音版本列表
}

struct DubbedAudioParams {                   # 配音请求参数（独立音频流）
    1: optional string LanguageCodeName = "" # 配音字幕标识码
    2: optional list<string> Versions        # 配音版本列表
}

struct VideoStyle {
    1: i64 Vstyle          #0-普通视频 1-VR视频
    2: i64 Dimension       #0-维度2D，1-维度3D上下，2-维度左右
    3: i64 ProjectionModel #投影模式 0-普通视频，1-等距柱状，2-cube map
    4: i64 ViewSize        #视野范围 0-普通视频，1-180度，2-360度
    5: i64 VRStreamingType  # 已经废弃
}

struct VideoMetaExtra {
  1: i64 bitrate
  2: i64 size
  3: string file_hash
  4: string file_id
}

struct VRView {
    1: string UriSuffix 				    # uri 后缀
    2: string InitRange					    # BaseRangeInfo
    3: string IndexRange 				    # BaseRangeInfo
    4: string CheckInfo  				    # CheckInfo
    5: list<double> Viewport 				    # 视角坐标
    6: VideoMetaExtra VideoMetaExtra 			    #视角特有属性
    7: optional string  FirstMoofRange                      #segmentBas firt moof range
}

struct DownloadFileParams {
    1: optional bool NeedDownload = false # 是否构建下载地址
    2: optional string Filename = ""      # 下载文件名字
}

struct FallbackApiRequestParams {
    1: optional string KeySeed
    2: optional string UrlEncryptMethod   # url加密方法
    3: optional bool   IsVideoModelV2Fallback = false #是否fallback到v2版本的video model
}

struct VideoExtraAttrs {
    1: optional string AudioType = "" # 音轨类型, dubbed_audio (配音), enhanced_audio (人声增强)
    2: optional string AudioTag = ""  # 人声增强削弱标记, enhance_70/enhance_90
}

struct BarrageMaskInfo {
    1: optional string Version        # 弹幕蒙板版本v1,v2...
    2: optional string BarrageMaskUrl # 弹幕蒙板url
    3: optional string FileId         # 弹幕蒙板文件Id
    4: optional i64 FileSize          # 弹幕蒙板文件大小
    5: optional string FileHash       # 弹幕蒙板文件哈希
    6: optional i64 UpdatedAt         # 弹幕蒙板文件更新日期
    7: optional i64 Bitrate           # 弹幕蒙板文件码率
    8: optional i64 HeadLen           # 弹幕蒙板文件头部大小
}

struct VideoHardDecodingInfo {
    1: optional string Profile
    2: optional string ColorTransfer
    3: optional string ColorRange
    4: optional i64    Level
    5: optional string ColorSpace
    6: optional string ColorPrimaries
}

struct GearInfo {
    1: optional string GearID           # 档位（组合）id
    2: optional string GearTag          # 档位（组合）tag，用于实验场景
    3: optional string GearSubID        # 该 VideoInfo 属于某个档位组合时，该字段不为空，值为该 VideoInfo 在组合内的唯一标识
    4: optional double UniversalVmaf    # universal_vmaf
    5: optional string GearName         # 档位名字
    6: optional double Crf              # crf
    7: optional string GearType         # 档位类型
}

struct VideoInfo {
   1: string MainUrl                                          #主播放地址
   2: string BackupUrl                                        #备播放地址
   3: Meta VideoMeta                                          #视频元信息
   4: optional PreloadInfo Preload                            #预加载信息 给点播sdk使用，其它业务可以忽略
   5: string PlayerAccessKey                                  #加密秘钥
   6: string MainHTTPUrl                                      #仅支持HTTP协议的主播放地址
   7: string BackupHTTPUrl                                    #仅支持HTTP协议的备播放地址
   8: i64 UrlExpire                                           #用于告诉客户端，当前返回的url过期时间（未来时间）
   9: optional string FileId                                  #视频File id，用于在p2p播放时的唯一标识
   10: optional string P2pVerifyUrl                           #p2p播放时crc文件所在地址
   11: optional string InitRange                              #segmentBase init range index
   12: optional string IndexRange                             #segmentBase index range
   13: optional string CheckInfo                              #video check info
   14: optional string BarrageMaskUrl                         #弹幕蒙版url
   15: optional string BarrageMaskOffset                      #弹幕蒙版offset
   16: optional FitterInfo Fitter                             #拟合函数
   17: string KeyframeAlignment                               #关键帧对齐
   18: optional string HelpInfoURL = ""                       #帮助信息地址
   19: string EncodeEID                                       #转码标识
   20: optional VolumeInfo VolumeInfo                         #平均音量
   21: optional map<string, list<SubtitleInfo>> SubtitleInfos #字幕信息
   22: optional i64 BizQualityType                            #业务转码流唯一标识
   23: optional string UserAction                             #视频架构下维护的状态
   24: optional string DubLanguageCodeName = ""               #配音语言标识码
   25: optional string DubVersion = ""                        #配音版本
   26: optional i64 DubLanguageID = 0                         #配音语言标识ID
   27: optional string EffectBarrageUrl                       #废弃该字段，到最外层取
   28: optional string AudioWaveUrl                           #音频波形图url
   29: optional VideoStyle VideoStyle                         #vr视频标识
   30: optional list<VRView> VRViewVideoList                  #vr多视角视频列表
   31: optional string PallasFidLabels = ""                   #画像标签json
   32: optional VideoExtraAttrs ExtraAttrs                    #视频/音频属性
   33: optional string  FirstMoofRange                        #segmentBas firt moof range
   34: optional VideoHardDecodingInfo  VHardDecodingInfo      #视频硬解码信息
   35: optional bool  PrivateURL = false                      #是否隐私URL
   36: optional GearInfo GearInfo                             #自适应档位组合
}

struct FilePlayInfo {
    1: i64 Status              #status 视频状态
    2: string Message               #同上
    3: optional VideoInfo VideoInfo      #转码视频信息,包括视频播放地址，视频元信息
    4: optional string HelpInfoURL = ""       # 帮助信息地址
    5: optional string VID = ""       # 视频ID
    6: optional string UserReference = ""         #item信息
}

struct Segment {
    1: double Begin
    2: double End
    3: double Speed
}

struct AISpeedInfo {
    1: string Version  //倍速版本
    2: optional string File  //格式化文件URL
    3: optional list<Segment> Segments  //倍速区间
}

struct DepthEstimation {
    1: double ConstDepth
}

struct SubtitlePosition {
    1: double X
    2: double Y
}

struct Luminance {
    1: double BrightRatio              // 亮度比例
    2: double Brightness                // 平均亮度
    3: double OverexposureRatio        // 过曝比例
    4: double StdBrightness            // 标准亮度
    5: double DiffOverexposureRatio   // 过曝光比例差值
}

struct PlayInfo {
    1: i64 Status                   #status 视频状态
    2: string Message               #同上
    3: string AccountName           #provider name
    4: string MediaType             #媒体类型 audio video
    5: double Duration              #原视频长度
    6: string PosterUrl             #封面截图
    7: optional SeekOffset Seekts
    8: optional list<BigThumb> BigThumbs
    9: optional map<string,list<string>> DnsInfo
    10: list<VideoInfo> VideoInfos                 #转码视频信息,包括视频播放地址，视频元信息
    11: optional VideoInfo OriginalVideoInfo       #片源视频信息,包括片源播放地址，视频元信息
    12: string PosterUri                           #封面截图uri
    13: i64 PopularityLevel                        #热度值
    14: optional string HelpInfoURL = ""           #帮助信息地址
    15: optional list<i64> LanguageIDs             #字幕语言列表
    16: optional VolumeInfo VolumeInfo             #平均音量
    17: optional SandwichInfo SandwichInfo         #伪横屏信息
    18: optional list<SubtitleInfo> SubtitleInfos  #字幕信息列表
    19: optional bool HasEmbeddedSubtitle          #是否有内嵌字幕
    20: optional string UserAction                 #视频架构下维护的状态
    21: map<string, list<VideoInfo>> DubInfos      #key是非配音流的FileID, 视频+音频混合流, 包括视频播放地址，视频元信息
    22: map<string, VideoInfo> DubbedAudioInfos    #key是dubbed_audio_n, 独立音频流, 包括视频播放地址，音频元信息
    23: optional string EffectBarrageUrl           #AI特效弹幕url
    24: optional list<AISpeedInfo> AiSpeedInfo     #视频智能倍速信息
    25: optional DepthEstimation DepthEstimation   #VR外挂字幕深度信息
    26: optional string PallasVidLabels = ""       #vid画像标签json
    27: optional string PallasDidLabels = ""       #用户画像标签json
    28: optional SubtitlePosition SubtitlePosition #VR外挂字幕位置
    29: optional Luminance Luminance               #亮度归一化
    30: optional string FullscreenStrategy = ""    #智能全屏策略
    31: optional string TitleInfo = ""             #文章标题信息
    32: optional string FullscreenMaxcrop = ""     #通顶视频裁剪
    33: optional BarrageMaskInfo BarrageMaskInfo   #蒙板弹幕
    #34已废弃，字段和国内的IDL有冲突
    35: optional string UserReference = ""         #item信息
    36: map<string, list<string>> PlayAddrDnsResult #DNS结果，用于冷启动优化
}

struct AuditVideoInfo {
    1: string IntranetUrl       #内外url（主）
    2: string ExtranetUrl       #外网url（主）
    3: Meta VideoMeta           #视频源信息
    4: i64 UrlExpire            #外网视频url有效期
    5: string PlayerAccessKey   #加密秘钥
    6: string FileId            #视频唯一file_id
    7: string BackupIntranetUrl #内外url（备）
    8: string BackupExtranetUrl #外外url（备）
}

struct AuditVideoPlayInfo {
    1: list<AuditVideoInfo> AuditVideoInfos //返回视频相关list
    2: AuditVideoInfo AuditOriginVideoInfo //片源视频的url信息
    3: string PosterUri // 封面uri
    4: i64 Status
    5: string Message
    6: optional string HelpInfoURL = ""    # 帮助信息地址
    7: optional VolumeInfo VolumeInfo  //平均音量
}

struct VideoPlayInfo {
    1: i64 Status                            #status 视频状态
    2: string Message                        #同上
    3: double VideoDuration                  #原视频长度
    4: string Validate                       #box头部添加vid信息,用于点播sdk,其它可以忽略
    5: bool EnableSSL                        #vid检验失败之后,开启https,同上
    6: map<string,VideoInfo> VideoInfos      #转码视频信息,包括视频播放地址，视频元信息
    7: map<string,string> OriginalVideoUrl   #片源视频内网播放地址
    8: list<BigThumb> BigThumbs
    9: map<string,list<string>> DnsInfo
    10: string poster_url                         #封面截图
    11: map<string,string> OriginalVideoCdnUrl    #片源视频cdn播放地址
    12: string AutoDefinition                     #端默认自动档分辨率。pgc
    13: optional VideoInfo OriginalVideoInfo      #片源视频信息,包括片源播放地址，视频元信息,注意：OriginalVideoInfo 新加字段，如果需要取源视频的相关信息，到这个字段中获取。
                                                  #除非request中指定外网地址，但是又想获取外网地址，则从上面的OriginalVideoCdnUrl,OriginalVideoUrl两个字段获取。
    14: optional SeekOffset Seekts
    15: string MediaType                           #媒体类型 audio video
    16: string AccountName                         #provider name
    17: string PosterUri                           #封面截图URI
    18: string OptimalDecodingMode                 #最优解码方式 hw/sw
    19: bool  EnableABRAdaptive                    #关键帧对齐
    20: string BarrageMaskUrl                      #弹幕蒙版URL
    21: string FullscreenStrategy                  #智能全屏策略
    22: optional VolumeInfo VolumeInfo             #平均音量
    23: i64 PopularityLevel                        #热度值
    24: optional string HelpInfoURL = ""           #帮助信息地址
    25: optional list<i64> LanguageIDs             #字幕语言列表
    26: optional SandwichInfo SandwichInfo         #伪横屏信息
    27: optional list<SubtitleInfo> SubtitleInfos  #字幕信息列表
    28: optional bool HasEmbeddedSubtitle          #是否有内嵌字幕
    29: optional string UserAction                 #视频架构下维护的状态
    30: map<string, list<VideoInfo>> DubInfos      #key是非配音流的FileID, 视频+音频混合流, 包括视频播放地址，视频元信息
    31: map<string, VideoInfo> DubbedAudioInfos    #key是dubbed_audio_n, 独立音频流, 包括视频播放地址，音频元信息
    32: optional string EffectBarrageUrl           #AI特效弹幕url
    33: optional list<AISpeedInfo> AiSpeedInfo     #下发视频智能倍速信息
    34: optional DepthEstimation DepthEstimation   #VR外挂字幕深度信息
    35: optional string PallasVidLabels = ""       #画像标签json
    36: optional string PallasDidLabels = ""       #画像标签json
    37: optional SubtitlePosition SubtitlePosition #VR外挂字幕位置
    38: optional Luminance Luminance               #亮度归一化
    39: optional string TitleInfo = ""
    40: optional string ResTag                     #资源标签,如付费视频等
    41: optional BarrageMaskInfo BarrageMaskInfo   #蒙版弹幕
    42: optional string FullscreenMaxcrop = ""     #通顶视频裁剪
    #43已废弃，字段和国内的IDL有冲突
    44: optional string UserReference = ""         #item信息
    45: map<string, list<string>> PlayAddrDnsResult #DNS结果，用于冷启动优化
}

struct MGetVideoPlayInfoRequest {
    1: required list<string> VIDs                                       # 视频ID列表
    2: optional UserInfo User                                           # 能传必须传过来，我们这边需要这个信息，进行cdn的调度，ab测试等
    3: optional VideoDefinition  NeedDefinition = VideoDefinition.ALL   # 指定分辨率(360p,480p,720p),若没有相应的分辨率,返回其它分辨率
    4: optional bool NeedBase64 = true                                  # CDN播放地址用base64编码,默认是true,目前内网地址,此参数不生效,都返回非base64编码
    5: optional VideoPlaySource  PlaySource = VideoPlaySource.Extranet  # 视频点播来源(内外网)
    6: optional VideoCodecType CodecType = VideoCodecType.H264          # 编码格式bytevc1 h264,默认是h264, 若没有bytevc1，则返回h264
    7: optional string LogoType = ""                                    # 水印名称,不明确制定,默认返回无logo的转码视频,
                                                                        # 若没有无logo的视频,返回带logo的视频,反之同样成立,fallback方案详见wiki
    8: optional string EncodedType = ""
    9: optional bool SSL = false
    10: optional string MagicCaller = ""                                #业务方无需关心
    11: optional VideoFormatType FormatType = VideoFormatType.Normal
    12: optional string PlayToken = ""                                  #分发token
    13: optional VideoCdnType CdnType = VideoCdnType.Normal             #端上支持的CDN类型
    14: optional SignInfo SignInfo                                      #废弃
    15: optional Identity Identity                                      # 身份验证相关参数
    16: optional bool NeedBarrageMask = false
    17: optional VideoQualityType VideoQuality = VideoQualityType.Normal# ies 720p 不同码率的视频，不传默认为正常码率，0为正常码率，1为低码率,2为高码率
    18: optional VideoEncryptionMethod EncryptionMethod = VideoEncryptionMethod.Normal  #加密模式
    19: optional list<VideoCdnType> CDNTypes # CDNType p2p 调度列表,如果非空，将优先使用CDNTypes调度，忽略CdnType
    20: optional bool NeedAiSpeed = false  #是否下发智能倍速
    21: optional bool EnableDeviceCodecControl = false #控制设备codec分发
    22: optional VideoDefinition  MaxDefinition = VideoDefinition.ALL #最高分辨率，NeedDefinition为ALL时生效，下发<=MaxDefinition的多个分辨率
    23: optional string AstraBusinessKeyInfo = "" #Astra 鉴权平台key
    24: optional bool NeedLuminance = false  #是否下发亮度归一化
    25: optional map<string, string> ForceFids
    26: optional bool ImmunePolicy = false  #是否豁免policy访问
    27: optional VodLibraConfig VodLibraConfig  # 业务透传点播libra实验
    28: optional map<string, string> BizTraceInfos                         #trace信息

    255: optional base.Base Base
}

struct GearIDWithFallback {
    1: optional list<string> GearIDs            # 档位（组合）id搜索列表
    2: optional string GearTag                  # 档位（组合）tag，gear_id_with_fallback不为空时有效
}

struct MGetPlayInfosRequest {
    1: required list<string> VIDs                                       # 视频ID列表
    2: optional UserInfo User                                           # 能传必须传过来,我们这边需要这个信息,进行cdn的调度,ab测试等
    3: optional VideoDefinition  NeedDefinition = VideoDefinition.ALL   # 指定分辨率(360p,480p,720p),若没有相应的分辨率,返回其它分辨率
    4: optional VideoPlaySource  PlaySource = VideoPlaySource.Extranet  # 视频点播来源(内外网)
    5: optional bool NeedBase64 = true                                  # CDN播放地址用base64编码,目前内网地址,此参数不生效,都返回非base64编码
    6: optional VideoCodecType CodecType = VideoCodecType.H264          # 编码格式bytevc1 h264,默认是h264, 若没有bytevc1，则返回h264
    7: optional string LogoType = ""                                    # 水印名称,不明确制定,默认返回无logo的转码视频,
                                                                        # 若没有无logo的视频,返回带logo的视频,反之同样成立
    8: optional string EncodedType = ""  #视频转码类型，默认""会返回普通非加密视频(normal). 有一下几种，按需传参
                                         #normal    普通非加密视频
                                         #audio     普通非加密音频
                                         #encrypt   加密视频
                                         #audio_encrypt    加密音频
                                         #normal_short 截断非加密视频

    9: optional VideoQualityType VideoQuality = VideoQualityType.Normal # ies 720p 不同码率的视频，不传默认为正常码率，0为正常码率，1为低码率,2为高码率
    10: optional bool SSL = false
    11: optional string MagicCaller = ""                                # 业务方无需关心
    12: optional VideoFormatType FormatType = VideoFormatType.Normal
    13: optional string PlayToken = ""                                  # 业务无需关心
    14: optional VideoCdnType CdnType = VideoCdnType.Normal             # 端上支持的CDN类型
    15: optional i64 Indate                                             # 有caller白名单，联系gaohonglei添加caller白名单。
                                                                        # 播放地址的有效期,单位s, indate= 3600 播放地址有效期就是1h
    16: optional string EncodeUserTag                                   # 转码时用户自定义的标签
    17: optional SignInfo SignInfo                                      # 废弃
    18: optional string RedirectUserAction                              # 对端机房302请求中视频状态
    19: optional string VLadder = ""                                    # 视频质量业务自定义档位
    20: optional bool NeedBarrageMask = false
    21: optional BigThumbVersion ThumbVersion = BigThumbVersion.V1
    22: optional string LibraVideoArchConfig                            # libra videoarch_biz_config
    23: optional map<string, BizScheduleScene> BizScheduleScenes        # 调度场景,业务侧透传冷热场景等
    24: optional Identity Identity                                      # 身份验证相关参数
    25: optional VideoHDRType HDRType = VideoHDRType.Normal             # HDR
    26: optional string BigThumbIMGParameter                            # 雪碧图图片处理参数
    27: optional map<string, list<SubtitleParams>> SubtitleParams       # key是videoID，value是字幕请求参数列表
    28: optional map<string, list<DubParams>> DubParams                 # key是videoID，value是配音请求参数列表
    29: optional VideoEncryptionMethod EncryptionMethod = VideoEncryptionMethod.Normal  #加密模式
    30: optional MpsOnlineTranscode OnlineTranscode                     # mps实时处理
    31: optional string ForceCodec = ""                                 # 避免设备codec自适应
    32: optional map<string, list<DubbedAudioParams>> DubbedAudioParams # key是videoID，value是配音请求参数列表
    33: optional list<VideoCdnType> CDNTypes # CDNType p2p 调度列表,如果非空，将优先使用CDNTypes调度，忽略CdnType
    34: optional bool NeedEffectBarrage = false           #AI特效弹幕
    35: optional PlayTimeLimit PlayTimeLimit  #播放起止时间
    36: optional bool NeedAudioWave = false   #是否需要音频波形图
    37: optional bool SelectiveEncodeUserTag = false  #是否额外返回用户标签为空的转码流
    38: optional map<string,CreatorInfo> CreatorInfos #创作者信息，key是vid，value是该视频的创作者信息 for mps online watermark
    39: optional bool NeedAiSpeed = false  #智能倍速信息
    40: optional bool EnableDeviceCodecControl = false #控制设备codec分发
    41: optional DownloadFileParams DownloadFileParams #视频下载参数
    42: optional ProjectionModelType ProjectionModelType = ProjectionModelType.Normal           #视频投影模式
    43: optional VideoDefinition  MaxDefinition = VideoDefinition.ALL #最高分辨率，NeedDefinition为ALL时生效，下发<=MaxDefinition的多个分辨率
    44: optional string AstraBusinessKeyInfo = "" #Astra 鉴权平台key
    45: optional bool NeedLuminance = false  #是否下发亮度归一化
    46: optional map<string, string> ForceFids  #指定fileID
    47: optional list<ProjectionModelType> ProjectionModelTypeList  #指定多个ProjectionModelType，如果非空，将比ProjectionModelType参数更高优
    48: optional bool ImmunePolicy = false  #是否豁免policy访问
    49: optional VodLibraConfig VodLibraConfig  # 业务透传点播libra实验
    50: optional map<string, string> BizTraceInfos                         #trace信息
    51: optional map<string, list<GearIDWithFallback>> Vid2GearIDWithFallbacks #自适应档位组合
    52: optional bool FreeFlowRequest = false  #是否是免流请求

    255: optional base.Base Base
}

enum PackPlayModelOptions {
    AtlasPlayModel = 0
    AtlasOriginalVideoInfo = 1
    AtlasTopRedirectURL = 2
    AtlasPlayRedirectURL = 3
    AtlasVodPlayModel = 4
    AtlasVodOriginalVideoInfo = 5
}

struct AtlasPlayModel {
    1: string AccountName
    2: string PlayModelJSON
    3: string OriginalVideoInfoJSON
    4: string TopRedirectURL
    5: string PlayRedirectURL
    6: string VodPlayModelJSON
    7: string VodOriginalVideoInfoJSON
    8: map<string,string> Extra
}

struct MGetHttpPlayModelRequest {
    1: required MGetPlayInfosRequest PlayInfosRequest
    2: optional PackPlayModelOptions PackOption = PackPlayModelOptions.AtlasPlayModel
    255: optional base.Base Base
}

struct MGetHttpPlayModelResponse {
     1: map<string, AtlasPlayModel> AtlasPlayModels
     255: optional base.BaseResp BaseResp
}

struct MGetAuditVideoInfoRequest {
    1: required list<string> VIDs
    2: optional VideoDefinition  NeedDefinition = VideoDefinition.ALL   # 指定分辨率(360p,480p,720p),若没有相应的分辨率,返回其它分辨率
    3: optional string EncodedType = "normal"  #视频转码类型有一下几种，按需传参
                                         # normal        普通非加密视频
                                         # audio         普通非加密音频
                                         # encrypt       加密视频
                                         # audio_encrypt 加密音频
                                         # normal_short  截断非加密视频
    4: optional i64 Indate               # 有caller白名单，联系gaohonglei添加caller白名单。
                                         # 播放地址的有效期,单位s, indate= 3600 播放地址有效期就是1h
    5: optional string AuditRegion       # 审核所在地区
    6: optional Identity Identity        # 身份验证相关参数
    7: optional i64 AppID
    8: optional VideoFormatType FormatType = VideoFormatType.Normal
    9: optional MpsOnlineTranscode OnlineTranscode #mps实时处理
    10: optional VideoEncryptionMethod EncryptionMethod = VideoEncryptionMethod.Normal  #加密模式
    11: optional PlayTimeLimit PlayTimeLimit  #播放起止时间
    12: optional string LogoType          #水印类型，pgc默认放出带水印视频，如果需要无水印，明确指定unwatermarked
    13: optional VideoDefinition  MaxDefinition = VideoDefinition.ALL #最高分辨率，NeedDefinition为ALL时生效，下发<=MaxDefinition的多个分辨率
    14: optional string AstraBusinessKeyInfo = "" #Astra 鉴权平台key
    15: optional string MagicCaller = ""
    16: optional string RedirectUserAction
    17: optional AuditUrlType  AuditUrlType  = AuditUrlType.AuditUrlType_InnerOuter # 打包urltype类型
    18: optional bool ImmunePolicy = false  #是否豁免policy访问
    19: optional VodLibraConfig VodLibraConfig  # 业务透传点播libra实验
    20: optional VideoCodecType CodecType = VideoCodecType.H264
    21: optional map<string, string> BizTraceInfos                         #trace信息
    22: optional VideoQualityType VQuality= VideoQualityType.Normal      # 码率质量类型
    23: optional string CountryCode = "unknown"                         #US IN等


    255: optional base.Base Base
}


struct MGetVideoInfoByFileIdRequest {
    1: required list<string> FileIds
    2: required UserInfo User           # 能传必须传过来,我们这边需要这个信息,进行cdn的调度,ab测试等
    3: optional UrlParams UrlParams # url控制参数，如https、地址类型、cdn类型、过期时间等
    4: optional map<string, string> FileIdVidMap  # 可选参数，fileId->vid 映射，提供该参数将忽略FileIds参数，通过vid查找videoinfo信息
    5: optional Identity Identity        # 身份验证相关参数
    6: optional map<string,CreatorInfo> CreatorInfos #创作者信息，key是vid，value是该视频的创作者信息 for mps online watermark
    7: optional VodLibraConfig VodLibraConfig  # 业务透传点播libra实验
    8: optional map<string, string> BizTraceInfos                         #trace信息

    255: optional base.Base Base
}

struct MGetVideoInfoByFileIdV2Request {
    1: required list<string> FileIds
    2: required UserInfo User           # 能传必须传过来,我们这边需要这个信息,进行cdn的调度,ab测试等
    3: optional UrlParams UrlParams # url控制参数，如https、地址类型、cdn类型、过期时间等
    4: optional map<string, string> FileIdVidMap  # 可选参数，fileId->vid 映射，提供该参数将忽略FileIds参数，通过vid查找videoinfo信息
    5: Identity Identity        # 身份验证相关参数
    6: optional map<string,CreatorInfo> CreatorInfos #创作者信息，key是vid，value是该视频的创作者信息 for mps online watermark transcode
    7: optional MpsOnlineTranscode OnlineTranscode                     # mps实时处理
    8: optional string AstraBusinessKeyInfo = "" #Astra 鉴权平台key
    9: optional FilterParams FilterParams
    10: optional VodLibraConfig VodLibraConfig  # 业务透传点播libra实验
    11: optional map<string, string> BizTraceInfos                         #trace信息
    12: optional bool FreeFlowRequest = false  #是否是免流请求

    255: optional base.Base Base
}

struct UnitTestOriginVideoInfo {
    1: required string UserAction
    2: required string EncodeStatus
    3: required string  CreatedAt
    4: required i64 Provider
    5: required string FileType
    6: required string VipPaid
    7: required string FileID
    8: optional string ProviderName
    9: optional i64    Width
    10: optional i64 Height
}

struct UnitTestEncodeVideoInfo {
    1: required string EncodedType
    2: required string FormatType
    3: required string CodecType
    4: required string Definition
    5: required string LogoType
    6: required string VQuality
    7: required string FileID
    8: optional string EncodeUserTag
}

struct UnitTestRequestInfo {
    1: required i64 AppID
    2: required string Format
    3: required string EncodedType
    4: required string CodecType
    5: required string LogoType
    6: required string Definition
    7: required string VQuality
    8: required string UserIP
    9: required string Action
    10: required string MagicCaller
    11: required string BizToken
    12: required i64 Vps
    13: optional string EncodeUserTag
    14: optional i64 DeviceID
}

struct UnitTestCase {
    1: required i64 ID
    2: required string Name
    3: required UnitTestRequestInfo RequestInfo
    4: required UnitTestOriginVideoInfo OriginVideo
    5: required list<UnitTestEncodeVideoInfo> EncodedVideos
    6: required i64 DesiredStatus
    7: required map<string,string> DesiredVideoList
}

struct ResultCase{
    1: required i64 ID
    2: required string Name
    3: required bool Pass
    4: required i64 Status
    5: required string ErrTag
    6: required map<string,string> VideoList
}

struct DoUnitTestRequest{
    1: required list<UnitTestCase> UnitTestCase

    255: optional base.Base Base
}

struct MGetVideoModelRequest {
    1: required list<string> VIDs
    2: optional FilterParams FilterParams                               # 过滤参数
    3: optional UserInfo User
    4: optional bool SSL = false                                        # 是否返回https
    5: optional i64 Indate                                              # 有caller白名单，联系gaohonglei添加caller白名单。
    6: optional string PlayToken = ""                                   # 复杂策略定制token，多数业务方无需关心
    7: optional VideoCdnType CdnType = VideoCdnType.Normal              # 过滤参数
    8: optional SignInfo SignInfo                                       # 已废弃
    9: optional bool NeedRefreshAPI = false                             # 下发403 fallback api
    10: optional Identity Identity                                      # 身份验证相关参数
    11: optional bool NeedBarrageMask = false
    12: optional bool NeedBase64 = false
    13: optional bool NeedBigThumbs = false
    14: optional bool NeedSeeks = false
    15: optional bool NeedDnsInfo = false
    16: optional bool NeedMultiRateAudios = false
    17: optional VideoModelVersion VideoModelVersion = VideoModelVersion.V1
    18: optional bool EnableDeviceCodecControl = false
    19: optional BigThumbVersion ThumbVersion = BigThumbVersion.V1
    20: optional string LibraVideoArchConfig                            # libra videoarch_biz_config
    21: optional map<string, BizScheduleScene> BizScheduleScenes        # 调度场景,业务侧透传冷热场景等
    22: optional UrlType UrlType = UrlType.VL0                          # 视频地址类型
    23: optional bool V1VideoModelNeedFitterInfo = false                # v1 版本videomodel是否需要拟合信息 fitter, 只在请求VideoModelVersion=V1时生效
    24: optional VideoHDRType HDRType = VideoHDRType.Normal
    25: optional string BigThumbIMGParameter                            # 雪碧图图片处理参数
    26: optional list<VideoCdnType> CDNTypes                            # CDNType p2p 调度列表,如果非空，将优先使用CDNTypes调度，忽略CdnType
    27: optional i64 UserRole = 0                                       # 字幕分发，用户角色 0（未知）1（作者）（已废弃）
    28: optional bool NeedEffectBarrage = false                         # AI特效弹幕
    29: optional PlayTimeLimit PlayTimeLimit                            # 播放起止时间
    30: optional bool SelectiveEncodeUserTag = false                    # 是否额外返回用户标签为空的转码流
    31: optional bool NeedAiSpeed = false                               # 是否需要智能倍速
    32: optional FallbackApiRequestParams  FallbackRequestParams        # fallback api请求参数
    33: optional string AstraBusinessKeyInfo = ""                       # Astra 鉴权平台key
    34: optional bool NeedLuminance = false                             # 是否下发亮度归一化
    35: optional map<string, string> ForceFids                          # 指定fileID
    36: optional string MagicCaller = ""                                # 业务方无需关心
    37: optional bool EnableDeviceDefinitionControl = false             # 根据设备性能控制最高分辨率的下发
    38: optional bool ImmunePolicy = false  #是否豁免policy访问
    39: optional bool NeedPktOffset = true                              # 是否需要pktoffset
    40: optional MpsOnlineTranscode OnlineTranscode                     # mps实时处理
    41: optional VodLibraConfig VodLibraConfig  # 业务透传点播libra实验
    42: optional map<string, string> BizTraceInfos                         #trace信息

    255: optional base.Base Base
}

struct VideoModelExtraVideoMeta {
    1: optional string Definition  #分辨率
    2: optional i64 FPS         #帧率
    3: optional string FileID   #文件ID
}

struct VideoModelExtraInfo {
    1: optional i64 Status
    2: optional string Message
    3: optional string LogoType
    4: optional VideoModelVersion VideoModelVersion
    5: optional string HelpInfoURL = ""             # 帮助信息地址
    6: optional i64 LengthOfVideoList
    7: optional bool IsDynamicVideo = false
    8: optional string UserAction = ""
    9: optional string AccountName = ""
    10: optional string DeniedVideoModelV1JSON = ""
    11: optional string ResTag = ""                 # 资源标签,如付费视频等
    12: optional string EncodeUserTag = ""
    13: optional map<string,VideoModelExtraVideoMeta> OBSOLETE_VideoMeta        #[已废弃]VideoModel中视频元信息，key是文件file_id(无序)
    14: optional map<string,VideoModelExtraVideoMeta> OBSOLETE_RemovedVideoMeta #[已废弃]不包含在VideoModel中转码流视频元信息,key是文件file_id(无序)
    15: optional list<VideoModelExtraVideoMeta> VideoMetaList          #VideoModel中视频元信息(按分辨率升序)
    16: optional list<VideoModelExtraVideoMeta> RemovedVideoMetaList   #不包含在VideoModel中转码流视频元信息(按分辨率升序)
    17: optional bool  PrivateURL = false          #是否隐私URL

}

struct MGetVideoModelResponse{
    1: map<string,string> VideoInfos
    2: map<string,VideoModelExtraInfo> VideoModelExtraInfos
    3: map<string,string> VideoInfosV2
    4: map<string,binary> PBVideoInfos
    5: optional string BaseHelpInfoURL = ""                      # 帮助信息地址

    255: optional base.BaseResp BaseResp
}

struct DoUnitTestResponse{
    1: required list<ResultCase> ResultCase

    255: optional base.BaseResp BaseResp
}

struct MGetVideoPlayInfoResponse {
    1: map<string,string> VideoInfos
    2: map<string,VideoModelExtraInfo> VideoModelExtraInfos
    3: optional string BaseHelpInfoURL = ""                          # 帮助信息地址

    255: optional base.BaseResp BaseResp
}

struct MGetPlayInfosResponse {
    1:   map<string, VideoPlayInfo> VideoInfos
    2:   optional string BaseHelpInfoURL = ""                          # 帮助信息地址
    255: optional base.BaseResp BaseResp
}

struct MGetAuditVideoInfoResponse {
    1: required map<string, AuditVideoPlayInfo> AuditVideoPlayInfos
    2: optional string BaseHelpInfoURL = ""                          # 帮助信息地址
    255: optional base.BaseResp BaseResp
}

struct MGetVideoInfoByFileIdV2Response {
    1: map<string, FilePlayInfo> VideoInfos
    2: optional string BaseHelpInfoURL = ""       # 帮助信息地址
    255: optional base.BaseResp BaseResp
}

struct MGetVideoInfoByFileIdResponse {
    1: map<string, VideoInfo> VideoInfos
    2: optional string BaseHelpInfoURL = ""        # 帮助信息地址
    255: optional base.BaseResp BaseResp
}

struct SetVideoStatusRequest{
    1: string VideoID      #视频ID
    2: VideoStatus Status  #视频可见度
    3: string IdentityInfo #身份认证信息

    255: base.Base Base
}

struct SetVideoStatusResponse{
    1: optional string BaseHelpInfoURL = ""       # 帮助信息地址
    255: base.BaseResp BaseResp
}

struct MGetPlayInfosV2Request {
    1: required list<string> VIDs   # 视频ID列表
    2: optional FilterParams FilterParams
    3: optional UserInfo User       # 能传必须传过来,我们这边需要这个信息,进行cdn的调度,ab测试等
    4: required UrlParams UrlParams # url控制参数，如https、地址类型、cdn类型、过期时间等
    5: required Identity Identity   # 身份验证相关参数
    6: optional bool NeedPreloadInfo = false       # 是否需要预加载信息
    7: optional bool NeedSeekOffset = true         # 是否需要seekOffset
    8: optional bool NeedBigThumbs = false         # 是否需要BigThumbs
    9: optional bool NeedOriginalVideoInfo = true  # 是否需要片源信息
    10: optional bool NeedDnsInfo = false
    11: optional bool NeedBarrageMask = false
    12: optional BigThumbVersion ThumbVersion = BigThumbVersion.V1
    13: optional string LibraVideoArchConfig                       # libra videoarch_biz_config
    14: optional map<string, BizScheduleScene> BizScheduleScenes   # 调度场景,业务侧透传冷热场景等
    15: optional string BigThumbIMGParameter                       # 雪碧图图片处理参数
    16: optional map<string, list<SubtitleParams>> SubtitleParams  # key是videoID，value是字幕请求参数列表
    17: optional map<string, list<DubParams>> DubParams            # key是videoID，value是配音请求参数列表
    18: optional MpsOnlineTranscode OnlineTranscode                # mps实时处理
    19: optional bool JSPlayer
    20: optional map<string, list<DubbedAudioParams>> DubbedAudioParams # key是videoID，value是配音请求参数列表
    21: optional bool NeedEffectBarrage = false           #AI特效弹幕
    22: optional bool NeedAudioWave = false               #是否需要音频波形图
    23: optional bool SelectiveEncodeUserTag = false #是否额外返回用户标签为空的转码流
    24: optional map<string,CreatorInfo> CreatorInfos #创作者信息，key是vid，value是该视频的创作者信息 for mps online watermark
    25: optional bool NeedAiSpeed = false  #智能倍速信息
    26: optional string PlayToken = ""               # 复杂策略定制token，多数业务方无需关心
    27: optional bool EnableDeviceCodecControl = false
    28: optional DownloadFileParams DownloadFileParams #视频下载参数
    29: optional string AstraBusinessKeyInfo = "" #Astra 鉴权平台key
    30: optional bool NeedLuminance = false  #是否下发亮度归一化
    31: optional map<string, string> ForceFids #指定fileID
    32: optional VideoHDRType HDRType = VideoHDRType.Normal
    33: optional bool NeedHardDecodingInfo = false # 默认不需要下发硬解码信息
    34: optional VodLibraConfig VodLibraConfig  # 业务透传点播libra实验
    35: optional map<string, list<GearIDWithFallback>> Vid2GearIDWithFallbacks #自适应档位组合
    36: optional map<string, string> BizTraceInfos                         #trace信息
    37: optional bool FreeFlowRequest = false  #是否是免流请求

    255: optional base.Base Base
}


struct MGetPlayInfosV2Response {
    1:   map<string, PlayInfo> VideoInfos
    2:   optional string BaseHelpInfoURL = ""       # 帮助信息地址
    255: optional base.BaseResp BaseResp
}

struct UrlInfo {
    1: required string MainUrl
    2: required string BackupUrl
    3: required i64 UrlExpire
    4: optional i64 status
    5: optional string HelpInfoURL = ""       # 帮助信息地址
}

struct MGetPlayUrlByObjectIDRequest {
    1: required list<string> ObjectIDs   # 视频ID列表
    3: optional UserInfo User       #应用ID
    4: required UrlParams UrlParams # url控制参数，如https、地址类型、cdn类型、过期时间等
    5: required Identity Identity   # 身份验证相关参数
    6: optional string Provider     # objects所属账号，TOB object必须传递
    7: optional string ContentType  # cdn地址下强制指定content-type类型
    8: optional string CdnScheduleScene #cdn调度场景
    9: optional VodLibraConfig VodLibraConfig  # 业务透传点播libra实验
    10: optional map<string, string> BizTraceInfos                         #trace信息
    11: optional string ResourceType #判断是否字幕自刷新请求

    255: optional base.Base Base
}

struct ObjectInfo{
    1: required string ObjectID
    2: required string VID
}

struct MGetPlayUrlByObjectInfoRequest {
    1: required list<ObjectInfo> ObjectInfos   # 视频ID列表
    3: optional UserInfo User       #应用ID
    4: required UrlParams UrlParams # url控制参数，如https、地址类型、cdn类型、过期时间等
    5: required Identity Identity   # 身份验证相关参数
    6: optional string ContentType  # cdn地址下强制指定content-type类型
    7: optional string CdnScheduleScene #cdn调度场景
    8: optional VodLibraConfig VodLibraConfig  # 业务透传点播libra实验
    9: optional map<string, string> BizTraceInfos                         #trace信息

    255: optional base.Base Base
}

struct MGetPlayUrlByObjectIDResponse {
    1:   map<string, UrlInfo> UrlInfos
    2:   optional string BaseHelpInfoURL = ""       # 帮助信息地址
    255: optional base.BaseResp BaseResp
}

struct MGetPlayUrlByObjectInfoResponse {
    1:   map<string, UrlInfo> UrlInfos
    2:   optional string BaseHelpInfoURL = ""       # 帮助信息地址
    255: optional base.BaseResp BaseResp
}

struct MetaInfo {
    1: optional bool HaveEncryptH264
    2: optional bool HaveEncryptByteVC1
    3: optional double Duration
    4: optional string Provider
    5: optional string HelpInfoURL = ""       # 帮助信息地址
    6: optional string Sandwich = "" # 伪横屏信息
    7: optional string UserAction
    8: i64 Status                   #status 视频状态
    9: string Message               #status Messagexin
}

struct MGetMetaInfoRequest {
    1: required list<string> VIDs
    2: optional string IdentityInfo             #身份认证信息，以SDK形式提供

    255: optional base.Base Base
}

struct MGetMetaInfoResponse {
    1:   map<string, MetaInfo> MetaInfos
    2:   optional string BaseHelpInfoURL = ""       # 帮助信息地址
    255: optional base.BaseResp BaseResp
}

struct MGetVideoMetaDataRequest {
    1: required list<string> VIDs
    2: required i64 TopAccountID
    3: required string SpaceName
    255: optional base.Base Base
}

struct MGetVideoMetaDataResponse {
    1: map<string,string> VideoMetaDatas
    2: string DataHash
    255: required base.BaseResp BaseResp
}

struct MGetMediaInfoRequest {
    1: required list<string> VIDs
    2: optional FilterParams FilterParams                               # 过滤参数
    3: optional UserInfo User
    4: optional bool SSL = false                                        # 是否返回https
    5: optional i64 Indate                                              # 有caller白名单，联系gaohonglei添加caller白名单。
    6: optional string PlayToken = ""                                   # 复杂策略定制token，多数业务方无需关心
    7: optional VideoCdnType CdnType = VideoCdnType.Normal              # 过滤参数
    8: optional Identity Identity                                       # 身份验证相关参数
    9: optional bool NeedBarrageMask = false
    10: optional bool NeedBase64 = false
    11: optional bool NeedBigThumbs = false
    12: optional bool NeedSeeks = false
    13: optional bool NeedDnsInfo = false
    14: optional bool NeedMultiRateAudios = false
    15: optional bool EnableDeviceCodecControl = false
    16: optional BigThumbVersion ThumbVersion = BigThumbVersion.V1
    17: optional string LibraVideoArchConfig                            # libra videoarch_biz_config
    18: optional map<string, BizScheduleScene> BizScheduleScenes        # 调度场景,业务侧透传冷热场景等
    19: optional UrlType UrlType = UrlType.VL0                          # 视频地址类型
    20: optional VideoHDRType HDRType = VideoHDRType.Normal
    21: optional string BigThumbIMGParameter                            # 雪碧图图片处理参数
    22: optional list<VideoCdnType> CDNTypes                            # CDNType p2p 调度列表,如果非空，将优先使用CDNTypes调度，忽略CdnType
    23: optional bool NeedEffectBarrage = false                         # AI特效弹幕
    24: optional PlayTimeLimit PlayTimeLimit                            # 播放起止时间
    25: optional bool SelectiveEncodeUserTag = false                    # 是否额外返回用户标签为空的转码流
    26: optional bool NeedAiSpeed = false                               # 是否需要智能倍速
    27: optional string AstraBusinessKeyInfo = ""                       # Astra 鉴权平台key
    28: optional bool NeedLuminance = false                             # 是否下发亮度归一化
    29: optional bool EnableDeviceDefinitionControl = false             # 根据设备性能控制最高分辨率的下发
    30: optional bool ImmunePolicy = false                              # 是否豁免policy访问
    31: optional bool NeedPktOffset = true                              # 是否需要pktoffset
    32: optional VodLibraConfig VodLibraConfig                          # 业务透传点播libra实验
    33: optional map<string, string> BizTraceInfos                      # trace信息


    255: optional base.Base Base
}

struct MediaInfo {
    1: string MainUrl
    2: string BackupUrl
    3: i64 UrlExpire
    4: string EncodedType
    5: string MediaType
    6: string LogoType
    7: string Definition
    8: string Quality
    9: string QualityDesc
    10: string Format
    11: i64 Width
    12: i64 Height
    13: i64 Bitrate
    14: i64 RealBitrate
    15: string CodecType
    16: i64 Size
    17: i64 FPS
    18: string FileId
    19: string FileHash
    20: string SubInfo
    21: i64 AvgBitrate
    22: optional string EncodeUserTag = ""
    23: optional string FidProfileLabels = ""
}

struct MediaResult {
    1: i64 Status
    2: string Message
    3: bool EnableSsl
    4: bool EnableAdaptive
    5: list<MediaInfo> MediaInfos
    6: optional list<BigThumb> BigThumbs
    7: optional string HelpInfoURL = ""
    8: optional BarrageMaskInfo BarrageMaskInfo
    9: optional VolumeInfo VolumeInfo
    10: double VideoDuration
    11: string ExtraInfo
    12: optional string VidProfileLabels = ""
    13: optional string DidProfileLabels = ""
}

struct MGetMediaInfoResponse {
    1:   map<string, MediaResult> MediaResults
    2:   optional string BaseHelpInfoURL = ""
    255: optional base.BaseResp BaseResp
}

struct MGetVVPlayInfosRequest {
    1: string VideoUrl
    2: i64 DownloadSpeed
    3: i64 IsColdStart
    4: i64 PreloadTaskPriority
    5: double PreloadTaskImportance
    6: string Version
    7: string LibraTag
    8: string UserIP
    9: i64 VideoUrlExpireTime

    255: optional base.Base Base
}

struct VVPlayScheduledUrl {
    1: string Url
    2: double QualityScore
    3: double CostScore
}

struct MGetVVPlayInfosResponse {
    1: VVPlayScheduledUrl OldUrl
    2: VVPlayScheduledUrl NewUrl

    255: optional base.BaseResp BaseResp
}

struct Item {
    1: required i64 ItemId
    2: required string VideoId
    3: required i32 AppId
    4: optional i32 Status
    5: optional i64 MediaType
    6: optional bool IsSpecialVr
    7: optional i64 CreateTime
    8: optional string stats      // ies.item.info stats
    9: optional string Content    // 参考 ies.item.info 的content字段,长视频则为longcontent
    10: optional string Extra     // 参考ies.item.info中的extra字段，至少需要传递：
                                  // is_ad，ad_source 及ad_type 3个字段
}

struct MGetAdaptivePlayInfoRequest {
    1: required list<string> VIDs
    2: required string TenantID
    3: required map<string, Item> ItemInfos
    4: optional FilterParams FilterParams                               # 过滤参数
    5: optional UserInfo User
    6: required UrlParams UrlParams                                     # url控制参数，如https、地址类型、cdn类型、过期时间等
    7: required Identity Identity                                       # 身份验证相关参数
    8: optional string PlayToken = ""                                   # 复杂策略定制token，多数业务方无需关心
    9: optional bool NeedBarrageMask = false
    10: optional bool NeedBase64 = false
    11: optional bool NeedBigThumbs = false
    12: optional BigThumbVersion ThumbVersion = BigThumbVersion.V1
    13: optional string LibraVideoArchConfig                            # libra videoarch_biz_config
    14: optional map<string, BizScheduleScene> BizScheduleScenes        # 调度场景,业务侧透传冷热场景等
    15: optional VideoHDRType HDRType = VideoHDRType.Normal
    16: optional bool SelectiveEncodeUserTag = false                    # 是否额外返回用户标签为空的转码流
    17: optional bool NeedLuminance = false                             # 是否下发亮度归一化
    18: optional map<string, list<SubtitleParams>> SubtitleParams       # key是videoID，value是字幕请求参数列表
    19: optional map<string, list<DubbedAudioParams>> DubbedAudioParams # key是videoID，value是配音请求参数列表
    20: optional MpsOnlineTranscode OnlineTranscode                     # mps实时处理
    21: optional map<string,CreatorInfo> CreatorInfos                   # 创作者信息，key是vid，value是该视频的创作者信息
    22: optional map<string, string> BizTraceInfos                      # trace信息
    23: optional map<string, list<GearIDWithFallback>> Vid2GearIDWithFallbacks #自适应档位组合
    24: optional bool FreeFlowRequest = false  #是否是免流请求
    25: optional string AdsVideoStrategy # 广告视频策略

    255: optional base.Base Base
}

struct AdaptivePlayMeta {
    1: required list<common.BitrateStruct> BitrateInfos      // 类型: pack/common.BitrateStruct
    2: required common.UrlStruct PlayAddr              // 类型: pack/common.UrlStruct
    3: optional common.UrlStruct PlayAddrPrevByteVC1   // 类型: pack/common.UrlStruct
    4: optional common.UrlStruct PlayAddrByteVC1       // pack/common.UrlStruct used for replace PrevByteVC1
    5: optional common.UrlStruct PlayAddrH264          // 类型: pack/common.UrlStruct
    6: required i32 IsPrevByteVC1
    7: optional string CodeType
    8: optional i64 Height
    9: optional i64 Width
    10: optional i32 Duration
    11: optional string Ratio
    12: optional i32 IsByteVC1
    13: optional string Meta                          // Video Meta Info, A map json string: map<string, string>
    14: optional string VidProfileLabels
    15: optional string DidProfileLabels
    16: map<string, list<string>> PlayAddrDnsResult
    17: optional list<common.BitrateAudioStruct> BitrateAudioInfos      // 类型: pack/common.BitrateAudioStruct
    18: optional string Format
}

struct AdaptivePlayInfo {
    1: AdaptivePlayMeta PlayMeta
    2: VideoPlayInfo PlayInfo
}

struct MGetAdaptivePlayInfoResponse {
    1:   map<string, AdaptivePlayInfo> AdaptivePlayInfos
    2:   optional string BaseHelpInfoURL = ""

    255: optional base.BaseResp BaseResp
}

service SmartPlayerService {
    # 旧接口, 返回json
    MGetVideoPlayInfoResponse MGetVideoPlayInfo(1:MGetVideoPlayInfoRequest request)
    # 新接口，返回thrift struct 建议新接入，都使用这个接口，返回的字段完全兼容旧接口
    MGetPlayInfosResponse MGetPlayInfos(1:MGetPlayInfosRequest request)
    # 根据FileId返回对应视频CDNUrl与P2pVerifyUrl，该接口用于p2p sdk在盒子预热视频做种
    MGetVideoInfoByFileIdResponse MGetVideoInfoByFileId(1:MGetVideoInfoByFileIdRequest request)
    # 新版的MGetVideoInfoByFileId接口
    MGetVideoInfoByFileIdV2Response MGetVideoInfoByFileIdV2(1:MGetVideoInfoByFileIdV2Request request)
    # 内网审核专用，返回特定的内外网域名,其他人慎用
    MGetAuditVideoInfoResponse MGetAuditPlayInfos(1:MGetAuditVideoInfoRequest request)
    # 单元测试专用接口
    DoUnitTestResponse DoUnitTest(1:DoUnitTestRequest request)
    # 点播sdk专用videoModel, 用于打包下发videoModel(播放地址/p2p信息/Dash信息)
    MGetVideoModelResponse MGetVideoModel(1:MGetVideoModelRequest request)
    # 变更视频状态
    SetVideoStatusResponse SetVideoStatus(1: SetVideoStatusRequest req)
    # 新版的MGetPlayInfos接口，强制校验身份认证信息，简化了响应结构
    MGetPlayInfosV2Response MGetPlayInfosV2(1: MGetPlayInfosV2Request request)
    # 根据storeURI生成播放地址
    MGetPlayUrlByObjectIDResponse MGetPlayUrlByObjectID(1: MGetPlayUrlByObjectIDRequest request)
    # 根据vid -> storeURI的映射生成播放地址
    MGetPlayUrlByObjectInfoResponse MGetPlayUrlByObjectInfo(1: MGetPlayUrlByObjectInfoRequest request)
    # 获取视频的基本信息
    MGetMetaInfoResponse MGetMetaInfo(1: MGetMetaInfoRequest request)
    # 废弃，业务不应该调用这个接口
    MGetVideoMetaDataResponse MGetVideoMetaData(1: MGetVideoMetaDataRequest request)
    # 点播atlas构建videoModel专用
    MGetHttpPlayModelResponse MGetHttpPlayModel(1: MGetHttpPlayModelRequest request)
    # 点播新版的videoModel接口，统一dash和mp4的响应结构
    MGetMediaInfoResponse MGetMediaInfo(1: MGetMediaInfoRequest request)
    # VV粒度播控接口
    MGetVVPlayInfosResponse MGetVVPlayInfos(1: MGetVVPlayInfosRequest request)
    # smart_player X VPP合并后的新接口
    MGetAdaptivePlayInfoResponse MGetAdaptivePlayInfo(1: MGetAdaptivePlayInfoRequest request)
}
