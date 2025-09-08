from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
from astrbot.api.star import Context, Star, register
from astrbot.api import logger, sp
from astrbot.api.all import *
from astrbot.core.message.components import Reply
from .utils.ttp import generate_image_openrouter
from .utils.file_send_server import send_file

@register("gemini-25-image-openrouter", "喵喵", "使用openrouter的免费api生成图片", "1.7")
class MyPlugin(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        # 支持多个API密钥
        self.openrouter_api_keys = config.get("openrouter_api_keys", [])
        # 向后兼容：如果还在使用旧的单个API密钥配置
        old_api_key = config.get("openrouter_api_key")
        if old_api_key and not self.openrouter_api_keys:
            self.openrouter_api_keys = [old_api_key]
        
        # 自定义API base支持 - 优先从配置文件加载，全局配置会在命令中覆盖
        self.custom_api_base = config.get("custom_api_base", "").strip()
        
        # 模型配置 - 优先从配置文件加载，全局配置会在命令中覆盖
        self.model_name = config.get("model_name", "google/gemini-2.5-flash-image-preview:free").strip()
        
        # 重试配置
        self.max_retry_attempts = config.get("max_retry_attempts", 3)
        
        self.nap_server_address = config.get("nap_server_address")
        self.nap_server_port = config.get("nap_server_port")
        
        # 标记是否已经加载过全局配置
        self._global_config_loaded = False

    async def _load_global_config(self):
        """异步加载全局配置"""
        if self._global_config_loaded:
            return
            
        try:
            plugin_config = await sp.global_get("gemini-25-image-openrouter", {})
            
            # 如果全局配置中有设置，则覆盖当前配置
            if "custom_api_base" in plugin_config:
                self.custom_api_base = plugin_config["custom_api_base"]
                logger.info(f"从全局配置加载 custom_api_base: {self.custom_api_base}")
                
            if "model_name" in plugin_config:
                self.model_name = plugin_config["model_name"]
                logger.info(f"从全局配置加载 model_name: {self.model_name}")
                
            self._global_config_loaded = True
        except Exception as e:
            logger.error(f"加载全局配置失败: {e}")
            self._global_config_loaded = True  # 即使失败也标记为已加载，避免重复尝试

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
        # 加载全局配置，确保使用最新的配置
        await self._load_global_config()
        
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
                api_base=self.custom_api_base if self.custom_api_base else None,
                max_retry_attempts=self.max_retry_attempts
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

    @filter.command_group("banana")
    def banan(self):
        """OpenRouter绘图插件快速配置命令组"""
        pass

    @banan.command("baseurl")
    async def switch_base_url(self, event: AstrMessageEvent, new_base_url: str = None, save_global: str = "false"):
        """快速切换openrouter绘图插件的base URL
        
        使用方法:
        /banan baseurl - 查看当前base URL
        /banan baseurl <新的base_url> - 临时切换base URL（会话级别）
        /banan baseurl <新的base_url> true - 永久切换base URL（全局配置）
        """
        # 确保加载最新的全局配置
        await self._load_global_config()
        
        if not new_base_url:
            current_url = self.custom_api_base if self.custom_api_base else "https://openrouter.ai/api/v1"
            yield event.plain_result(f"当前 base URL: {current_url}\n使用方法:\n/banan baseurl <新的base_url> - 临时切换\n/banan baseurl <新的base_url> true - 永久保存")
            return
        
        # 更新当前实例的配置
        self.custom_api_base = new_base_url.strip()
        
        # 根据 save_global 参数决定是否保存到全局配置
        if save_global.lower() in ["true", "1", "yes", "y"]:
            try:
                # 获取插件的全局配置
                plugin_config = await sp.global_get("gemini-25-image-openrouter", {})
                plugin_config["custom_api_base"] = new_base_url.strip()
                await sp.global_put("gemini-25-image-openrouter", plugin_config)
                yield event.plain_result(f"已永久切换 base URL 到: {new_base_url}（已保存到全局配置）")
            except Exception as e:
                logger.error(f"保存全局配置失败: {e}")
                yield event.plain_result(f"已临时切换 base URL 到: {new_base_url}（保存全局配置失败: {str(e)}）")
        else:
            yield event.plain_result(f"已临时切换 base URL 到: {new_base_url}（会话级别，重启后恢复）")

    @banan.command("model")
    async def switch_model(self, event: AstrMessageEvent, new_model: str = None, save_global: str = "false"):
        """快速切换openrouter绘图插件的模型
        
        使用方法:
        /banan model - 查看当前模型
        /banan model <模型名> - 临时切换模型（会话级别）
        /banan model <模型名> true - 永久切换模型（全局配置）
        
        例如: /banan model google/gemini-2.5-flash-image-preview:free
        """
        # 确保加载最新的全局配置
        await self._load_global_config()
        
        if not new_model:
            yield event.plain_result(f"当前模型: {self.model_name}\n使用方法:\n/banan model <模型名> - 临时切换\n/banan model <模型名> true - 永久保存\n例如: /banan model google/gemini-2.5-flash-image-preview:free")
            return
        
        # 更新当前实例的配置，使用用户输入的完整准确的模型名
        self.model_name = new_model.strip()
        
        # 根据 save_global 参数决定是否保存到全局配置
        if save_global.lower() in ["true", "1", "yes", "y"]:
            try:
                # 获取插件的全局配置
                plugin_config = await sp.global_get("gemini-25-image-openrouter", {})
                plugin_config["model_name"] = new_model.strip()
                await sp.global_put("gemini-25-image-openrouter", plugin_config)
                yield event.plain_result(f"已永久切换模型到: {new_model}（已保存到全局配置）")
            except Exception as e:
                logger.error(f"保存全局配置失败: {e}")
                yield event.plain_result(f"已临时切换模型到: {new_model}（保存全局配置失败: {str(e)}）")
        else:
            yield event.plain_result(f"已临时切换模型到: {new_model}（会话级别，重启后恢复）")
