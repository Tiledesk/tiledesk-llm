"""
Image Semantic Linker
Links images to text descriptions and generates semantic metadata.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class ImageSemanticLinker:
    """
    Links images to their context in the document and generates
    enhanced semantic metadata.
    """

    def __init__(self, graph_repository: Optional[Any] = None):
        self.graph_repository = graph_repository

    async def link_image_to_context(
        self,
        image_data: Dict[str, Any],
        text_elements: List[Dict[str, Any]],
        llm: Any,
        doc_id: str
    ) -> Dict[str, Any]:
        """
        Enhance image metadata by linking to surrounding text and generating captions.
        """
        image_id = image_data.get('id')
        
        # 1. Find text referencing this image
        referencing_elements = self._find_references(image_id, text_elements)
        ref_texts = [e.get('text', '') for e in referencing_elements]
        context_text = " ".join(ref_texts)
        
        image_data['surrounding_text'] = context_text
        
        # Captioning is usually handled in logic.py via generate_image_caption
        # which already takes context_text.

        # 2. Update Knowledge Graph if repository is available
        if self.graph_repository:
            try:
                from tilellm.modules.knowledge_graph.models import Relationship
                
                # Link referencing paragraphs to image
                for elem in referencing_elements:
                    rel = Relationship(
                        source_id=elem.get('id'),
                        target_id=image_id,
                        type="REFERENCES",
                        properties={"doc_id": doc_id}
                    )
                    self.graph_repository.create_relationship(rel)
            except Exception as e:
                logger.error(f"Error updating graph for image {image_id}: {e}")

        return image_data

    def _find_references(self, image_id: str, text_elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find text elements that reference the image/figure ID or number."""
        import re
        
        # Simple heuristic: extract number from ID
        try:
            # doc_id_image_page_index
            image_num = image_id.split('_')[-1]
        except:
            image_num = None
            
        refs = []
        patterns = [rf"Figure\s+{image_num}\b", rf"Fig\.?\s+{image_num}\b"]
        if not image_num:
             patterns = [r"Figure\s+\d+", r"Fig\.?\s+\d+"]

        for elem in text_elements:
            text = elem.get('text', '')
            for p in patterns:
                if re.search(p, text, re.IGNORECASE):
                    refs.append(elem)
                    break
                    
        return refs
