import logging
import os
import time
from pathlib import Path
import euler
import requests
import thriftpy2
from servicediscovery import ServiceDiscoveryError
import base64
import codecs
import hmac
import time
from hashlib import sha1
from decord import VideoReader, cpu
import io

AUTH_PREFIX_V1 = "VARCH1-HMAC-SHA1"
DEFAULT_TTL = 3600  # in seconds
SEP = ":"

def sign_rpc_request(ak, sk, method='', caller='', extra=None, ttl=0):
    if not extra:
        extra = {}
    if ttl <= 0:
        ttl = DEFAULT_TTL
    deadline = str(int(time.time()) + ttl)

    arr = ['method=' + method, 'caller=' + caller, 'deadline=' + deadline]
    arr.extend([k + '=' + extra[k] for k in sorted(extra.keys())])

    raw = '&'.join(arr)
    hashed = hmac.new(codecs.encode(sk), codecs.encode(raw), sha1)
    dig = hashed.digest()
    ciphertext = base64.standard_b64encode(dig)
    return SEP.join([AUTH_PREFIX_V1, ak, deadline, codecs.decode(ciphertext)])


def sign_http_request(ak, sk, method, url, ttl=0, required=None):
    if not required:
        required = {}
    from urllib.parse import parse_qsl, urlparse

    q = urlparse(url)
    m = method + q.path
    caller = ''
    extraInQuery = dict(parse_qsl(q.query))
    extra = {}
    for key, value in extraInQuery.items():
        if required.get(key, False):
            extra[key] = value
    return sign_rpc_request(ak, sk, method=m, caller=caller, extra=extra, ttl=ttl)


logger = logging.getLogger(__name__)
class VideoUrlNotFound(Exception):
    pass
class Client:
    def __init__(self, ak, sk, scene, caller='tiktok.aiic.data', idc='maliva', cluster='default'):
        self.ak = ak
        self.sk = sk
        self.scene = scene
        self.method_name = 'MGetPlayInfosV2'
        self.caller = caller
        idl_path = os.path.join(Path(__file__).parent, 'idl')
        self.smart_player = thriftpy2.load(
            os.path.join(idl_path, 'videoarch/smart_player.thrift'), module_name='smart_player_thrift'
        )
        self.base = thriftpy2.load(os.path.join(idl_path, 'base.thrift'), module_name='base_thrift')
        self.client = euler.Client(
            self.smart_player.SmartPlayerService,
            f'sd://toutiao.videoarch.smart_player?idc={idc}&cluster={cluster}',
            timeout=10,
        )
    def wait_sd(self, timeout=None):
        """
        等待服务发现就绪，刚启动任务直接调用通常因为服务发现问题失败，需先调用此方法等待服务就绪
        :param timeout: 等待超时, 超时还未就绪则抛出异常, 单位 s
        """
        start = time.perf_counter()
        while True:
            try:
                self.mget_video_url([])
            except ServiceDiscoveryError as e:
                logger.warning(f'service discovery not ready: {e}')
            except Exception as e:
                logger.info(f'it is not service discovery error, it ready to call. {e}')
                return
            if timeout and time.perf_counter() - start > timeout:
                raise TimeoutError()
            time.sleep(0.1)
    def get_video_stream(self, vid, url_hint=None, need_definition=6, chunk_size=8192):
        """
        获取视频 bytes 迭代器，获取不到 url 会抛出 VideoUrlNotFound 异常
        :param vid: 视频 vid
        :param url_hint: 若指定了 url，则直接从 url 中读取视频
        :param need_definition: 指定分辨率，默认 540p，选项见：https://bytedance.larkoffice.com/wiki/wikcnlEBKqakxSFFAj2L70mBtmb
        :param chunk_size: 流式获取字节，一个 chunk 的大小
        """
        if url_hint:
            yield from self._get_video_stream_from_url(url_hint, chunk_size)
            return
        urls = self.mget_video_url([vid], need_definition)
        if vid in urls:
            yield from self._get_video_stream_from_url(urls[vid], chunk_size)
            return
        raise VideoUrlNotFound(f'video {vid} url not found')
    def get_video_bytes(self, vid, url_hint=None, need_definition=6):
        """
        获取视频 bytes，获取不到 url 会抛出 VideoUrlNotFound 异常
        :param vid: 视频 vid
        :param url_hint: 若指定了 url，则直接从 url 中读取视频
        :param need_definition: 指定分辨率，默认 540p，选项见：https://bytedance.larkoffice.com/wiki/wikcnlEBKqakxSFFAj2L70mBtmb
        """
        if url_hint:
            return self._get_video_bytes_from_url(url_hint)
        urls = self.mget_video_url([vid], need_definition)
        if vid in urls:
            return self._get_video_bytes_from_url(urls[vid])
        raise VideoUrlNotFound(f'video {vid} url not found')
    def mget_video_url(self, vids, indate=3600, url_type=9, need_definition=6):
        """
        批量获取视频 url，会过滤掉审核不通过、封禁、不存在、不可播放等状态的视频，返回格式 dict[vid, url]
        :param vids: 视频 vid 列表
        :param indate: url 有效期，单位秒
        :param url_type: url 类型，默认 9 内网研发网络。10 为办公网络可访问
        :param need_definition: 指定分辨率，默认 540p，选项见：https://bytedance.larkoffice.com/wiki/wikcnlEBKqakxSFFAj2L70mBtmb
        """
        sig = sign_rpc_request(self.ak, self.sk, self.method_name, self.caller, {})
        req = self.smart_player.MGetPlayInfosV2Request(
            VIDs=vids,
            User=self.smart_player.UserInfo(AppID=1180),
            Identity=self.smart_player.Identity(IdentityInfo=sig),
            UrlParams=self.smart_player.UrlParams(UrlType=url_type, Indate=indate),
            FilterParams=self.smart_player.FilterParams(Watermark='unwatermarked', NeedDefinition=need_definition),
            NeedOriginalVideoInfo=True,
            Base=self.base.Base(Caller=self.caller),
        )
        resp = self.client.MGetPlayInfosV2(req)
        if resp.BaseResp and resp.BaseResp.StatusCode != 0:
            raise Exception(
                f'invoke smart player err, code: {resp.BaseResp.StatusCode}, msg: {resp.BaseResp.StatusMessage}'
            )
        res = {}
        for vid, info in resp.VideoInfos.items():
            if info.Status != 10:
                logger.warning(f'video {vid} status {info.Status} is not normal, skip')
                continue
            if len(info.VideoInfos) == 0:
                logger.warning(f'video {vid} info empty, skip')
                continue
            res[vid] = info.VideoInfos[0].MainUrl
        return res
    def _get_video_stream_from_url(self, url, chunk_size):
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            yield from r.iter_content(chunk_size=chunk_size)
    def _get_video_bytes_from_url(self, url):
        with requests.get(url) as r:
            r.raise_for_status()
            return r.content

