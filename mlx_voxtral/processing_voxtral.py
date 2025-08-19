import numpy as np
import mlx.core as mx
from typing import Optional, Dict, Union, List, Any, Tuple
from transformers import AutoTokenizer

from .audio_processing import VoxtralFeatureExtractor


class VoxtralProcessor:
    """Processor that combines audio feature extraction and text tokenization."""

    def __init__(
        self,
        feature_extractor=None,
        tokenizer=None,
    ):
        self.feature_extractor = feature_extractor or VoxtralFeatureExtractor()
        self.tokenizer = tokenizer
        
        if tokenizer is not None:
            self._special_token_ids = self._get_special_token_ids()
        else:
            self._special_token_ids = None
    
    def _get_special_token_ids(self) -> Dict[str, int]:
        """Get special token IDs from tokenizer."""
        special_tokens = {}
        
        if hasattr(self.tokenizer, 'get_control_token'):
            special_tokens['bos'] = self.tokenizer.get_control_token('<s>')
            special_tokens['eos'] = self.tokenizer.get_control_token('</s>')
            special_tokens['inst'] = self.tokenizer.get_control_token('[INST]')
            special_tokens['inst_end'] = self.tokenizer.get_control_token('[/INST]')
            special_tokens['audio'] = self.tokenizer.get_control_token('[AUDIO]')
            special_tokens['begin_audio'] = self.tokenizer.get_control_token('[BEGIN_AUDIO]')
            special_tokens['transcribe'] = self.tokenizer.get_control_token('[TRANSCRIBE]')
        elif hasattr(self.tokenizer, 'audio_token_id'):
            special_tokens['audio'] = self.tokenizer.audio_token_id
            if hasattr(self.tokenizer, 'vocab'):
                vocab = self.tokenizer.vocab
                special_tokens['bos'] = vocab.get('<s>', 1)
                special_tokens['eos'] = vocab.get('</s>', 2)
                special_tokens['inst'] = vocab.get('[INST]', 3)
                special_tokens['inst_end'] = vocab.get('[/INST]', 4)
                special_tokens['begin_audio'] = vocab.get('[BEGIN_AUDIO]', 25)
                special_tokens['transcribe'] = vocab.get('[TRANSCRIBE]', 34)
        else:
            special_tokens = {
                'bos': 1,
                'eos': 2,
                'inst': 3,
                'inst_end': 4,
                'audio': 24,
                'begin_audio': 25,
                'transcribe': 34,
            }
        
        if hasattr(self.tokenizer, 'pad_token_id'):
            special_tokens['pad'] = self.tokenizer.pad_token_id
        else:
            special_tokens['pad'] = 0 
            
        return special_tokens


    def __call__(
        self,
        text: Optional[str] = None,
        audio: Optional[Union[np.ndarray, List[float], str]] = None,
        sampling_rate: int = 16000,
        padding: bool = True,
        **kwargs,
    ) -> Dict[str, mx.array]:
        """Process audio and text inputs."""

        encoding = {}

        if audio is not None:
            audio_features = self.feature_extractor(
                audio,
                sampling_rate=sampling_rate,
                return_tensors="mlx",  
            )
            encoding["input_features"] = audio_features["input_features"]


        if text is not None and self.tokenizer is not None:

            if hasattr(self.tokenizer, "__call__"):
                text_encoding = self.tokenizer(
                    text, return_tensors="mlx", padding=padding, **kwargs
                )
                encoding["input_ids"] = text_encoding["input_ids"]
                if "attention_mask" in text_encoding:
                    encoding["attention_mask"] = text_encoding["attention_mask"]
            else:
                token_ids = self.tokenizer.encode(text)
                if len(token_ids) > 0 and token_ids[-1] == 2:  # 2 is EOS
                    token_ids = token_ids[:-1]
                if len(token_ids) > 1 and token_ids[0] == 1 and token_ids[1] == 1:
                    token_ids = token_ids[1:]

                encoding["input_ids"] = mx.array([token_ids])
                if padding:
                    encoding["attention_mask"] = mx.ones_like(encoding["input_ids"])

            if audio is not None and self._special_token_ids is not None:
                # Find and replace audio placeholder sequences
                # "<audio>" can tokenize in different ways:
                # - [8175, 9383, 1062] = ["<a", "udio", ">"] when standalone
                # - [1534, 57401, 1062] = [" <", "audio", ">"] when preceded by space
                audio_placeholder_sequences = [
                    [8175, 9383, 1062],  # "<audio>"
                    [1534, 57401, 1062],  # " <audio>"
                ]

                new_input_ids = []
                for batch_idx in range(encoding["input_ids"].shape[0]):
                    batch_ids = encoding["input_ids"][batch_idx]
                    ids_list = batch_ids.tolist()

                    found = False
                    for audio_placeholder_sequence in audio_placeholder_sequences:
                        for i in range(
                            len(ids_list) - len(audio_placeholder_sequence) + 1
                        ):
                            if (
                                ids_list[i : i + len(audio_placeholder_sequence)]
                                == audio_placeholder_sequence
                            ):
                                # Found the sequence! Replace it with 375 audio tokens
                                # Remove the placeholder sequence
                                for _ in range(len(audio_placeholder_sequence)):
                                    ids_list.pop(i)
                                # Insert 375 audio tokens
                                audio_token_id = self._special_token_ids.get('audio', 24)
                                for _ in range(375):
                                    ids_list.insert(i, audio_token_id)
                                found = True
                                break
                        if found:
                            break

                    new_input_ids.append(mx.array(ids_list))

                    if found and "attention_mask" in encoding:
                        encoding["attention_mask"] = mx.ones(
                            (encoding["input_ids"].shape[0], len(ids_list))
                        )

                if new_input_ids:
                    max_len = max(len(ids) for ids in new_input_ids)
                    padded_ids = []
                    for ids in new_input_ids:
                        if len(ids) < max_len:
                            pad_token_id = self._special_token_ids.get('pad', 0)
                            padding = [pad_token_id] * (max_len - len(ids))
                            padded_ids.append(mx.concatenate([ids, mx.array(padding)]))
                        else:
                            padded_ids.append(ids)
                    encoding["input_ids"] = mx.stack(padded_ids)

        return encoding

    def batch_decode(self, token_ids, **kwargs):
        """Decode token IDs to text."""
        if self.tokenizer is None:
            raise ValueError("No tokenizer provided")

        if isinstance(token_ids, mx.array):
            token_ids = np.array(token_ids)

        return self.tokenizer.batch_decode(token_ids, **kwargs)

    def decode(self, token_ids, **kwargs):
        """Decode token IDs to text."""
        if self.tokenizer is None:
            raise ValueError("No tokenizer provided")

        if isinstance(token_ids, mx.array):
            token_ids = np.array(token_ids)

        if hasattr(self.tokenizer, "decode") and callable(self.tokenizer.decode):
            return self.tokenizer.decode(token_ids, **kwargs)
        else:
            if hasattr(token_ids, "tolist"):
                token_ids = token_ids.tolist()
            return self.tokenizer.decode(token_ids)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load processor from pretrained model."""
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )

        feature_extractor = VoxtralFeatureExtractor()

        return cls(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
        )

    def apply_transcrition_request(
        self,
        audio: Union[str, np.ndarray, List[float]],
        language: Optional[str] = None,
        sampling_rate: Optional[int] = None,
    ):
        """Apply the model's transcription request template.

        This formats the input according to Voxtral's expected format:
        [INST][BEGIN_AUDIO][AUDIO]...[/INST]lang:{language}[TRANSCRIBE]
        
        If language is None, the format is:
        [INST][BEGIN_AUDIO][AUDIO]...[/INST][TRANSCRIBE]

        Args:
            audio: Audio input - can be:
                - URL string (http:// or https://)
                - File path string
                - NumPy array
                - List of floats
            language: Language code (e.g. "en"). If None, no language specification is added.
            sampling_rate: Sampling rate (default: 16000)
        """
        # Process audio (feature extractor handles URLs, file paths, arrays, etc.)
        audio_features = self.feature_extractor(
            audio,
            sampling_rate=sampling_rate if sampling_rate else 16000,
            return_tensors="mlx",
        )

        # Create the transcription prompt
        # The format is: <s>[INST][BEGIN_AUDIO][AUDIO tokens][/INST]lang:{language}[TRANSCRIBE]
        
        if self._special_token_ids is None:
            raise ValueError("Tokenizer is required for apply_transcrition_request")

        # Start with BOS token
        tokens = [self._special_token_ids['bos']]

        # Add [INST] token
        tokens.append(self._special_token_ids['inst'])

        # Add [BEGIN_AUDIO]
        tokens.append(self._special_token_ids['begin_audio'])

        # Calculate number of audio tokens needed
        # Each 30-second chunk needs 375 tokens
        if isinstance(audio_features, dict):
            input_features = audio_features["input_features"]
        else:
            input_features = audio_features.input_features
        num_chunks = input_features.shape[0]
        num_audio_tokens = num_chunks * 375

        # Add [AUDIO] tokens
        tokens.extend([self._special_token_ids['audio']] * num_audio_tokens)

        # Add [/INST]
        tokens.append(self._special_token_ids['inst_end'])

        # Add language specification if provided
        if language is not None:
            # Tokenize "lang:{language}" properly
            lang_str = f"lang:{language}"
            if hasattr(self.tokenizer, 'encode'):
                # Encode without special tokens
                lang_tokens = self.tokenizer.encode(lang_str, add_special_tokens=False)
                if isinstance(lang_tokens, mx.array):
                    lang_tokens = lang_tokens.tolist()
                tokens.extend(lang_tokens)
            else:
                # Fallback for specific languages
                if language == "en":
                    tokens.extend([9909, 1058, 1262])  # "lang:en"
                else:
                    raise NotImplementedError(f"Language {language} not yet supported")

        tokens.append(self._special_token_ids['transcribe'])

        input_ids = mx.array([tokens], dtype=mx.int32)

        class TranscriptionInputs:
            def __init__(self, input_ids, input_features):
                self.input_ids = input_ids
                self.input_features = input_features

            def to(self, device, dtype=None):
                return self

        return TranscriptionInputs(input_ids, input_features)
    
    def _parse_message_content(
        self, content: Union[str, List[Dict[str, Any]]]
    ) -> Tuple[List[Dict[str, Any]], List[Union[str, np.ndarray]]]:
        """Parse message content into structured format and extract audio data.
        
        Returns:
            content_items: List of content items with type and data
            audio_data: List of audio data (paths, arrays, etc.)
        """
        if isinstance(content, str):
            return [{"type": "text", "text": content}], []
        
        content_items = []
        audio_data = []
        
        for item in content:
            if item["type"] == "text":
                content_items.append(item)
            elif item["type"] == "audio":
                content_items.append(item)
                if "audio" in item:
                    audio_data.append(item["audio"])
                elif "path" in item:
                    audio_data.append(item["path"])
                elif "url" in item:
                    audio_data.append(item["url"])
                elif "base64" in item:
                    audio_data.append(item["base64"])
                else:
                    raise ValueError("Audio content must have 'audio', 'path', 'url', or 'base64' field")
            else:
                raise ValueError(f"Unknown content type: {item['type']}")
        
        return content_items, audio_data
    
    def apply_chat_template(
        self,
        conversation: Union[List[Dict[str, Any]], Dict[str, Any]],
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        return_tensors: Optional[str] = None,
        **kwargs,
    ) -> Union[str, Dict[str, mx.array]]:
        """Apply chat template to format conversation messages.
        
        Args:
            conversation: Single message dict or list of message dicts with 'role' and 'content'
            tokenize: If True, return tokenized output; if False, return formatted string
            add_generation_prompt: If True, add prompt for assistant response
            return_tensors: "mlx" for MLX arrays, None for lists
            
        Returns:
            Formatted string if tokenize=False, otherwise dict with:
            - input_ids: Token IDs as MLX array
            - input_features: Audio features as MLX array (if audio present)
            - attention_mask: Attention mask (optional)
        """
        if self._special_token_ids is None:
            raise ValueError("Tokenizer is required for apply_chat_template")
        
        if isinstance(conversation, dict):
            conversation = [conversation]
        
        all_tokens = []
        all_audio_features = []
        
        all_tokens.append(self._special_token_ids['bos'])
        
        for i, message in enumerate(conversation):
            role = message["role"]
            content = message["content"]
            
            content_items, audio_data = self._parse_message_content(content)
            
            if role == "system":
                # System message format: system\n{content}</s>
                all_tokens.extend(self.tokenizer.encode("system\n", add_special_tokens=False))
                
                for item in content_items:
                    if item["type"] == "text":
                        text_tokens = self.tokenizer.encode(item["text"], add_special_tokens=False)
                        all_tokens.extend(text_tokens)
                
                all_tokens.append(self._special_token_ids['eos'])
                
            elif role == "user":
                # User message format: [INST]{content}[/INST]
                all_tokens.append(self._special_token_ids['inst'])
                
                for item in content_items:
                    if item["type"] == "text":
                        text_tokens = self.tokenizer.encode(item["text"], add_special_tokens=False)
                        all_tokens.extend(text_tokens)
                    elif item["type"] == "audio":
                        audio_idx = len(all_audio_features)
                        
                        if "audio" in item:
                            audio_input = item["audio"]
                        elif "path" in item:
                            audio_input = item["path"]
                        elif "url" in item:
                            audio_input = item["url"]
                        elif "base64" in item:
                            raise NotImplementedError("Base64 audio not yet supported")
                        
                        sampling_rate = item.get("sampling_rate", None)
                        
                        audio_features = self.feature_extractor(
                            audio_input,
                            sampling_rate=sampling_rate,
                            return_tensors="mlx",
                        )
                        
                        if isinstance(audio_features, dict):
                            features = audio_features["input_features"]
                        else:
                            features = audio_features.input_features
                        
                        all_audio_features.append(features)
                        
                        all_tokens.append(self._special_token_ids['begin_audio'])
                        num_chunks = features.shape[0]
                        num_audio_tokens = num_chunks * 375
                        all_tokens.extend([self._special_token_ids['audio']] * num_audio_tokens)
                
                all_tokens.append(self._special_token_ids['inst_end'])
                
                if i == len(conversation) - 1 and len(content_items) == 1 and content_items[0]["type"] == "audio":
                    lang_str = "lang:en"  # Default to English
                    lang_tokens = self.tokenizer.encode(lang_str, add_special_tokens=False)
                    all_tokens.extend(lang_tokens)
                    all_tokens.append(self._special_token_ids['transcribe'])
                
            elif role == "assistant":
                for item in content_items:
                    if item["type"] == "text":
                        text_tokens = self.tokenizer.encode(item["text"], add_special_tokens=False)
                        all_tokens.extend(text_tokens)
                
                if not (i == len(conversation) - 1 and add_generation_prompt):
                    all_tokens.append(self._special_token_ids['eos'])
            
            else:
                raise ValueError(f"Unknown role: {role}")
        
        if not tokenize:
            if hasattr(self.tokenizer, 'decode'):
                return self.tokenizer.decode(all_tokens)
            else:
                return "<formatted string not available>"
        
        output = {}
        
        if return_tensors == "mlx":
            output["input_ids"] = mx.array([all_tokens], dtype=mx.int32)
            
            if all_audio_features:
                output["input_features"] = mx.concatenate(all_audio_features, axis=0)
            
            output["attention_mask"] = mx.ones_like(output["input_ids"])
        else:
            output["input_ids"] = [all_tokens]
            
            if all_audio_features:
                output["input_features"] = all_audio_features
            
            output["attention_mask"] = [[1] * len(all_tokens)]
        
        return output
