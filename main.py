from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
from astrbot.api.all import *
from astrbot.core.message.components import Reply
from .utils.ttp import generate_image_openrouter
from .utils.file_send_server import send_file

@register("gemini-25-image-openrouter", "喵喵", "使用openrouter的免费api生成图片", "1.5")
class MyPlugin(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        # 支持多个API密钥
        self.openrouter_api_keys = config.get("openrouter_api_keys", [])
        # 向后兼容：如果还在使用旧的单个API密钥配置
        old_api_key = config.get("openrouter_api_key")
        if old_api_key and not self.openrouter_api_keys:
            self.openrouter_api_keys = [old_api_key]
        
        # 自定义API base支持
        self.custom_api_base = config.get("custom_api_base", "").strip()
        
        # 模型配置
        self.model_name = config.get("model_name", "google/gemini-2.5-flash-image-preview:free").strip()
        
        self.nap_server_address = config.get("nap_server_address")
        self.nap_server_port = config.get("nap_server_port")

    async def send_image_with_callback_api(self, image_path: str) -> Image:
        """
        优先使用callback_api_base发送图片，失败则退回到本地文件发送
        
        Args:
            image_path (str): 图片文件路径
            
        Returns:
            Image: 图片组件
        """
        callback_api_base = self.context.get_config().get("callback_api_base")
        if not callback_api_base:
            logger.info("未配置callback_api_base，使用本地文件发送")
            return Image.fromFileSystem(image_path)

        logger.info(f"检测到配置了callback_api_base: {callback_api_base}")
        try:
            image_component = Image.fromFileSystem(image_path)
            download_url = await image_component.convert_to_web_link()
            logger.info(f"成功生成下载链接: {download_url}")
            return Image.fromURL(download_url)
        except (IOError, OSError) as e:
            logger.warning(f"文件操作失败: {e}，将退回到本地文件发送")
            return Image.fromFileSystem(image_path)
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"网络连接失败: {e}，将退回到本地文件发送")
            return Image.fromFileSystem(image_path)
        except Exception as e:
            logger.error(f"发送图片时出现未预期的错误: {e}，将退回到本地文件发送")
            return Image.fromFileSystem(image_path)

    @llm_tool(name="gemini-pic-gen")
    async def pic_gen(self, event: AstrMessageEvent, image_description: str, use_reference_images: bool = True):
        """
            Generate or modify images using the Gemini model via the OpenRouter API.
            When a user requests image generation or drawing, call this function.
            If use_reference_images is True and the user has provided images in their message,
            those images will be used as references for generation or modification.
            If no images are provided or use_reference_images is False, pure text-to-image generation will be performed.

            Here are some examples:
            1. If the user wants to generate a large figure model, such as an anime character with normal proportions, please use a prompt like:
            "Please accurately transform the main subject in this photo into a realistic, masterpiece-like 1/7 scale PVC statue.
            A box should be placed beside the statue: the front of the box should have a large, clear transparent window printed with the main artwork, product name, brand logo, barcode, and a small specification or authenticity verification panel. A small price tag sticker must also be attached to the corner of the box. Meanwhile, a computer monitor should be placed at the back, and the monitor screen needs to display the ZBrush modeling process of this statue.
            In front of the packaging box, the statue should be placed on a round plastic base. The statue must have 3D dimensionality and a sense of realism, and the texture of the PVC material needs to be clearly represented. If the background can be set as an indoor scene, the effect will be even better.

            Below are detailed guidelines to note:
            When repairing any missing parts, there must be no poorly executed elements.
            When repairing human figures (if applicable), the body parts must be natural, movements must be coordinated, and the proportions of all parts must be reasonable.
            If the original photo is not a full-body shot, try to supplement the statue to make it a full-body version.
            The human figure's expression and movements must be exactly consistent with those in the photo.
            The figure's head should not appear too large, its legs should not appear too short, and the figure should not look stunted—this guideline may be ignored if the statue is a chibi-style design.
            For animal statues, the realism and level of detail of the fur should be reduced to make it more like a statue rather than the real original creature.
            No outer outline lines should be present, and the statue must not be flat.
            Please pay attention to the perspective relationship of near objects appearing larger and far objects smaller."

            2. If the user wants to generate a chibi figure model or a small, cute figure, please use a prompt like:
            "Please accurately transform the main subject in this photo into a realistic, masterpiece-like 1/7 scale PVC statue.
            Behind the side of this statue, a box should be placed: on the front of the box, the original image I entered, with the themed artwork, product name, brand logo, barcode, and a small specification or authenticity verification panel. A small price tag sticker must also be attached to one corner of the box. Meanwhile, a computer monitor should be placed at the back, and the monitor screen needs to display the ZBrush modeling process of this statue.
            In front of the packaging box, the statue should be placed on a round plastic base. The statue must have 3D dimensionality and a sense of realism, and the texture of the PVC material needs to be clearly represented. If the background can be set as an indoor scene, the effect will be even better.

            Below are detailed guidelines to note:
            When repairing any missing parts, there must be no poorly executed elements.
            When repairing human figures (if applicable), the body parts must be natural, movements must be coordinated, and the proportions of all parts must be reasonable.
            If the original photo is not a full-body shot, try to supplement the statue to make it a full-body version.
            The human figure's expression and movements must be exactly consistent with those in the photo.
            The figure's head should not appear too large, its legs should not appear too short, and the figure should not look stunted—this guideline may be ignored if the statue is a chibi-style design.
            For animal statues, the realism and level of detail of the fur should be reduced to make it more like a statue rather than the real original creature.
            No outer outline lines should be present, and the statue must not be flat.
            Please pay attention to the perspective relationship of near objects appearing larger and far objects smaller."

            Args:
            - image_description (string): Description of the image to generate. Translate to English can be better.
            - use_reference_images (bool): Whether to use images from the user's message as reference. Default True.
        """
        openrouter_api_keys = self.openrouter_api_keys
        nap_server_address = self.nap_server_address
        nap_server_port = self.nap_server_port

        # 根据参数决定是否使用参考图片
        input_images = []
        if use_reference_images:
            # 从当前对话上下文中获取图片信息
            if hasattr(event, 'message_obj') and event.message_obj and hasattr(event.message_obj, 'message'):
                for comp in event.message_obj.message:
                    if isinstance(comp, Image):
                        try:
                            base64_data = await comp.convert_to_base64()
                            input_images.append(base64_data)
                        except (IOError, ValueError, OSError) as e:
                            logger.warning(f"转换当前消息中的参考图片到base64失败: {e}")
                        except Exception as e:
                            logger.error(f"处理当前消息中的图片时出现未预期的错误: {e}")
                    elif isinstance(comp, Reply):
                        # 修复引用消息中的图片获取逻辑
                        # Reply组件的chain字段包含被引用的消息内容
                        if comp.chain:
                            for reply_comp in comp.chain:
                                if isinstance(reply_comp, Image):
                                    try:
                                        base64_data = await reply_comp.convert_to_base64()
                                        input_images.append(base64_data)
                                        logger.info(f"从引用消息中获取到图片")
                                    except (IOError, ValueError, OSError) as e:
                                        logger.warning(f"转换引用消息中的参考图片到base64失败: {e}")
                                    except Exception as e:
                                        logger.error(f"处理引用消息中的图片时出现未预期的错误: {e}")
                        else:
                            logger.debug("引用消息的chain为空，无法获取图片内容")
            
            # 记录使用的图片数量
            if input_images:
                logger.info(f"使用了 {len(input_images)} 张参考图片进行图像生成")
            else:
                logger.info("未找到参考图片，执行纯文本图像生成")

        # 调用生成图像的函数
        try:
            image_url, image_path = await generate_image_openrouter(
                image_description,
                openrouter_api_keys,
                model=self.model_name,
                input_images=input_images,
                api_base=self.custom_api_base if self.custom_api_base else None
            )
            
            if not image_url or not image_path:
                # 生成失败，发送错误消息
                error_chain = [Plain("图像生成失败，请检查API配置和网络连接。")]
                yield event.chain_result(error_chain)
                return
            
            # 处理文件传输和图片发送
            if self.nap_server_address and self.nap_server_address != "localhost":
                image_path = await send_file(image_path, HOST=nap_server_address, PORT=nap_server_port)
            
            # 使用新的发送方法，优先使用callback_api_base
            image_component = await self.send_image_with_callback_api(image_path)
            chain = [image_component]
            yield event.chain_result(chain)
            return
                
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"网络连接错误导致图像生成失败: {e}")
            error_chain = [Plain(f"网络连接错误，图像生成失败: {str(e)}")]
            yield event.chain_result(error_chain)
            return
        except ValueError as e:
            logger.error(f"参数错误导致图像生成失败: {e}")
            error_chain = [Plain(f"参数错误，图像生成失败: {str(e)}")]
            yield event.chain_result(error_chain)
            return
        except Exception as e:
            logger.error(f"图像生成过程出现未预期的错误: {e}")
            error_chain = [Plain(f"图像生成失败: {str(e)}")]
            yield event.chain_result(error_chain)
            return
