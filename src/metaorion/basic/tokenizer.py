#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project : MetaIndex 
# @File    : tokenizer.py
# @Author  : zhangchao
# @Date    : 2024/12/23 14:16 
# @Email   : zhangchao5@genomics.cn
import os
import sentencepiece as spm
from typing import Dict, Union, List

from transformers import PreTrainedTokenizer


class MetaOrionTokenizer(PreTrainedTokenizer):
    def __init__(self, model_name_or_path, **kwargs):
        self.model = spm.SentencePieceProcessor()
        self.model.load(model_file=os.path.join(model_name_or_path, 'tokenizer.model'))
        super().__init__(
            pad_token='<pad>',
            unk_token='<unk>',
            bos_token='<s>',
            eos_token='</s>',
            cls_token='<cls>',
            mask_token='<mask>',
        )
        self.word_separator = '▁'

    @property
    def vocab_size(self) -> int:
        return self.model.get_piece_size()

    def get_vocab(self) -> Dict[str, int]:
        return {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}

    def _tokenize(self, text: str) -> List[str]:
        return [self.word_separator + x for x in text.split(' ')]

    def _convert_token_to_id(self, token):
        return self.model.piece_to_id(token)

    def _convert_id_to_token(self, index: int) -> str:
        return self.model.id_to_piece(index)

    def _decode(
            self,
            token_ids: Union[int, List[int]],
            skip_special_tokens: bool = False,
            clean_up_tokenization_spaces: bool = None,
            spaces_between_special_tokens: bool = True,
            **kwargs,
    ) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        text = self.model.decode(token_ids)
        return text
