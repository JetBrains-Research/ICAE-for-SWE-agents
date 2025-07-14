"""constants"""

from typing import List, Dict

class TemplateManager:
    """Centralized template management for ICAE models.
    
    This class provides methods to construct all the different template patterns
    used throughout the codebase, ensuring consistency and single-point maintenance.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.template_tokens = self._get_template_tokens()

    def _get_template_tokens(self) -> Dict[str, List[int]]:
        """Get template tokens from tokenizer. This is the core function that defines how text templates are created."""
        # Get the full template with generation prompt to extract all parts
        full_template = self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': '{{content}}'}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        
        # Split by the content placeholder
        user_prefix_str, rest_str = full_template.split('{{content}}')
        
        assistant_start_marker = '<|im_start|>assistant\n'
        suffix_str, think_str = rest_str.split(assistant_start_marker, 1)
        
        assistant_prefix_str = assistant_start_marker + think_str

        tokenized_parts = {
            'user_prefix': self.tokenizer.encode(user_prefix_str, add_special_tokens=False),
            'assistant_prefix': self.tokenizer.encode(assistant_prefix_str, add_special_tokens=False),
            'suffix': self.tokenizer.encode(suffix_str, add_special_tokens=False)
        }

        return tokenized_parts

    def _safe_decode_with_mem_tokens(self, token_ids: List[int]) -> str:
        """Decodes token IDs, replacing None with 'MEMTOKEN' for special tokens."""
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        return self.tokenizer.convert_tokens_to_string([t if t is not None else "MEMTOKEN" for t in tokens])

    def _apply_chat_template(self, content: List[int]) -> List[int]:
        """Applies the chat template structure around the content.
        Pattern: <user_prefix> + content + <suffix> + <assistant_prefix>"""
        return (
            self.template_tokens['user_prefix'] +
            content +
            self.template_tokens['suffix'] +
            self.template_tokens['assistant_prefix']
        )

    def create_squad_prompt(self, memory_tokens: List[int], question_tokens: List[int]) -> List[int]:
        """Pattern: <user_prefix> + memory_tokens + question + <suffix> + <assistant_prefix>"""
        return self._apply_chat_template(memory_tokens + question_tokens)

    def create_encoder_input(self, content_tokens: List[int]) -> List[int]:
        """Pattern: <user_prefix> + content + <suffix> + <assistant_prefix>"""
        return self._apply_chat_template(content_tokens)
    
    def create_decoder_prompt_ae(self, memory_tokens: List[int], ae_token_id: int) -> List[int]:
        """Pattern: <user_prefix> + memory_tokens + [ae_token] + <suffix> + <assistant_prefix>"""
        return self._apply_chat_template(memory_tokens + [ae_token_id])
    
    def create_decoder_prompt_lm(self, memory_tokens: List[int]) -> List[int]:
        """Pattern: <user_prefix> + memory_tokens + <suffix> + <assistant_prefix>"""
        return self._apply_chat_template(memory_tokens)
    
    def create_answer_with_suffix(self, answer_tokens: List[int]) -> List[int]:
        """Pattern: answer + <suffix>"""
        return answer_tokens + self.template_tokens['suffix']
