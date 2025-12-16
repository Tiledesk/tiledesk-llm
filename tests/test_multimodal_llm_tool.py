"""
Unit tests per il tool multimodale interno

Questi test verificano il corretto funzionamento del tool multimodale
che consente all'LLM di invocare se stesso per operazioni multimodali.
"""

import unittest
import asyncio
import base64
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any

# Import del tool da testare
from tilellm.tools.multimodal_llm_tool import (
    create_multimodal_llm_tool,
    create_provider_specific_multimodal_tool,
    MultimodalLLMInput
)


class TestMultimodalLLMTool(unittest.TestCase):
    """Test suite per il tool multimodale interno"""

    def setUp(self):
        """Setup eseguito prima di ogni test"""
        # Mock dell'istanza LLM
        self.mock_llm = AsyncMock()
        self.mock_llm.bind = Mock(return_value=self.mock_llm)

        # Mock della risposta LLM
        self.mock_response = Mock()
        self.mock_response.content = "Questa è una risposta di test dall'LLM"
        self.mock_llm.ainvoke = AsyncMock(return_value=self.mock_response)

    def test_create_multimodal_tool(self):
        """Test: Creazione del tool multimodale"""
        tool = create_multimodal_llm_tool(self.mock_llm)

        # Verifica che il tool sia stato creato correttamente
        self.assertIsNotNone(tool)
        self.assertEqual(tool.name, "invoke_multimodal_llm")
        self.assertIn("multimodal", tool.description.lower())

        # Verifica che il tool abbia lo schema corretto
        self.assertTrue(hasattr(tool, 'args_schema'))

    def test_tool_schema(self):
        """Test: Verifica dello schema del tool"""
        tool = create_multimodal_llm_tool(self.mock_llm)

        # VECCHIO CODICE (commentato):
        # Verifica semplice dell'esistenza dello schema
        # self.assertTrue(hasattr(tool, 'args_schema'))

        # NUOVO CODICE: Verifica dettagliata dello schema
        schema = tool.args_schema
        self.assertEqual(schema, MultimodalLLMInput)

        # Verifica i campi dello schema
        schema_fields = schema.model_fields
        self.assertIn('prompt', schema_fields)
        self.assertIn('images_base64', schema_fields)
        self.assertIn('documents_base64', schema_fields)
        self.assertIn('system_prompt', schema_fields)
        self.assertIn('max_tokens', schema_fields)
        self.assertIn('temperature', schema_fields)

    def test_invoke_tool_text_only(self):
        """Test: Invocazione del tool con solo testo"""
        tool = create_multimodal_llm_tool(self.mock_llm)

        # Esegui il tool
        result = asyncio.run(tool.ainvoke({
            "prompt": "Analizza questo testo",
            "max_tokens": 512,
            "temperature": 0.0
        }))

        # Verifica che l'LLM sia stato chiamato
        self.mock_llm.ainvoke.assert_called_once()

        # Verifica il risultato
        self.assertEqual(result, "Questa è una risposta di test dall'LLM")

        # Verifica che bind sia stato chiamato con i parametri corretti
        self.mock_llm.bind.assert_called_with(max_tokens=512, temperature=0.0)

    def test_invoke_tool_with_images(self):
        """Test: Invocazione del tool con immagini"""
        tool = create_multimodal_llm_tool(self.mock_llm)

        # Crea immagini di test in base64
        test_image_1 = base64.b64encode(b"fake_image_data_1").decode('utf-8')
        test_image_2 = base64.b64encode(b"fake_image_data_2").decode('utf-8')

        # Esegui il tool
        result = asyncio.run(tool.ainvoke({
            "prompt": "Cosa vedi in queste immagini?",
            "images_base64": [test_image_1, test_image_2]
        }))

        # Verifica che l'LLM sia stato chiamato
        self.mock_llm.ainvoke.assert_called_once()

        # Verifica che il messaggio contenga le immagini
        call_args = self.mock_llm.ainvoke.call_args[0][0]
        self.assertEqual(len(call_args), 2)  # SystemMessage + HumanMessage

        # Verifica il contenuto del messaggio umano
        human_message = call_args[1]
        content = human_message.content

        # Deve contenere 1 testo + 2 immagini = 3 elementi
        self.assertEqual(len(content), 3)
        self.assertEqual(content[0]['type'], 'text')
        self.assertEqual(content[1]['type'], 'image')
        self.assertEqual(content[2]['type'], 'image')

    def test_invoke_tool_with_documents(self):
        """Test: Invocazione del tool con documenti"""
        tool = create_multimodal_llm_tool(self.mock_llm)

        # Crea documento di test in base64
        test_pdf = base64.b64encode(b"fake_pdf_data").decode('utf-8')

        # Esegui il tool
        result = asyncio.run(tool.ainvoke({
            "prompt": "Riassumi questo documento",
            "documents_base64": [{
                "data": test_pdf,
                "mime_type": "application/pdf"
            }]
        }))

        # Verifica che l'LLM sia stato chiamato
        self.mock_llm.ainvoke.assert_called_once()

        # Verifica che il messaggio contenga il documento
        call_args = self.mock_llm.ainvoke.call_args[0][0]
        human_message = call_args[1]
        content = human_message.content

        # Deve contenere 1 testo + 1 documento = 2 elementi
        self.assertEqual(len(content), 2)
        self.assertEqual(content[0]['type'], 'text')
        self.assertEqual(content[1]['type'], 'document')
        self.assertEqual(content[1]['source']['media_type'], 'application/pdf')

    def test_invoke_tool_multimodal_complex(self):
        """Test: Invocazione del tool con contenuto multimodale complesso"""
        tool = create_multimodal_llm_tool(self.mock_llm)

        # Prepara contenuti misti
        test_image = base64.b64encode(b"fake_image").decode('utf-8')
        test_doc = base64.b64encode(b"fake_doc").decode('utf-8')

        # Esegui il tool
        result = asyncio.run(tool.ainvoke({
            "prompt": "Analizza questa immagine e questo documento insieme",
            "images_base64": [test_image],
            "documents_base64": [{
                "data": test_doc,
                "mime_type": "application/pdf"
            }],
            "system_prompt": "You are an expert analyst",
            "max_tokens": 1024,
            "temperature": 0.3
        }))

        # Verifica la chiamata
        self.mock_llm.ainvoke.assert_called_once()
        self.mock_llm.bind.assert_called_with(max_tokens=1024, temperature=0.3)

        # Verifica il contenuto del messaggio
        call_args = self.mock_llm.ainvoke.call_args[0][0]
        self.assertEqual(len(call_args), 2)  # System + Human

        # Verifica system message personalizzato
        system_message = call_args[0]
        self.assertIn("expert analyst", system_message.content)

        # Verifica contenuto multimodale: 1 text + 1 image + 1 document = 3
        human_message = call_args[1]
        content = human_message.content
        self.assertEqual(len(content), 3)

    def test_invoke_tool_with_data_uri_prefix(self):
        """Test: Il tool gestisce correttamente i prefissi data: URI"""
        tool = create_multimodal_llm_tool(self.mock_llm)

        # Immagine con prefisso data URI (come potrebbe arrivare da alcuni client)
        test_image = "data:image/jpeg;base64," + base64.b64encode(b"image").decode('utf-8')

        # Esegui il tool
        result = asyncio.run(tool.ainvoke({
            "prompt": "Analizza questa immagine",
            "images_base64": [test_image]
        }))

        # Verifica che il prefisso sia stato rimosso
        call_args = self.mock_llm.ainvoke.call_args[0][0]
        human_message = call_args[1]
        content = human_message.content

        # L'immagine non deve contenere il prefisso data:
        image_data = content[1]['source']['data']
        self.assertNotIn('data:', image_data)
        self.assertNotIn('base64,', image_data)

    def test_invoke_tool_error_handling(self):
        """Test: Gestione degli errori durante l'invocazione"""
        tool = create_multimodal_llm_tool(self.mock_llm)

        # Simula un errore nell'LLM
        self.mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM Error"))

        # Esegui il tool
        result = asyncio.run(tool.ainvoke({
            "prompt": "Test error handling"
        }))

        # Verifica che l'errore sia stato gestito
        self.assertIn("ERROR", result)
        self.assertIn("LLM Error", result)

    def test_create_provider_specific_tool(self):
        """Test: Creazione di tool specifici per provider"""

        # Test per diversi provider
        providers = ["openai", "anthropic", "google"]

        for provider in providers:
            tool = create_provider_specific_multimodal_tool(provider, self.mock_llm)

            # Verifica che il tool sia stato creato
            self.assertIsNotNone(tool)
            self.assertEqual(tool.name, "invoke_multimodal_llm")

    def test_tool_with_empty_images_list(self):
        """Test: Il tool gestisce correttamente liste vuote"""
        tool = create_multimodal_llm_tool(self.mock_llm)

        # Esegui con liste vuote (non None)
        result = asyncio.run(tool.ainvoke({
            "prompt": "Solo testo",
            "images_base64": [],
            "documents_base64": []
        }))

        # Verifica che funzioni come solo testo
        call_args = self.mock_llm.ainvoke.call_args[0][0]
        human_message = call_args[1]
        content = human_message.content

        # Solo 1 elemento di testo
        self.assertEqual(len(content), 1)
        self.assertEqual(content[0]['type'], 'text')

    def test_multimodal_input_validation(self):
        """Test: Validazione dello schema MultimodalLLMInput"""

        # Input valido
        valid_input = MultimodalLLMInput(
            prompt="Test prompt",
            max_tokens=512,
            temperature=0.5
        )
        self.assertEqual(valid_input.prompt, "Test prompt")
        self.assertEqual(valid_input.max_tokens, 512)
        self.assertEqual(valid_input.temperature, 0.5)

        # Verifica valori di default
        self.assertIsNone(valid_input.images_base64)
        self.assertIsNone(valid_input.documents_base64)
        self.assertIsNotNone(valid_input.system_prompt)


class TestMultimodalToolIntegration(unittest.TestCase):
    """Test di integrazione per il tool multimodale con MCP"""

    @patch('tilellm.tools.multimodal_llm_tool.HumanMessage')
    @patch('tilellm.tools.multimodal_llm_tool.SystemMessage')
    def test_tool_message_construction(self, mock_system_msg, mock_human_msg):
        """Test: Verifica costruzione corretta dei messaggi"""

        # VECCHIO CODICE (commentato):
        # Test semplice senza mock dei messaggi

        # NUOVO CODICE: Test con mock per verificare i parametri esatti
        mock_llm = AsyncMock()
        mock_llm.bind = Mock(return_value=mock_llm)
        mock_response = Mock()
        mock_response.content = "Test response"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        tool = create_multimodal_llm_tool(mock_llm)

        # Esegui il tool
        asyncio.run(tool.ainvoke({
            "prompt": "Test",
            "system_prompt": "Custom system"
        }))

        # Verifica che i messaggi siano stati costruiti
        mock_system_msg.assert_called_once()
        mock_human_msg.assert_called_once()

    def test_realistic_workflow(self):
        """Test: Simula un workflow realistico con MCP"""

        # Simula uno scenario reale:
        # 1. Tool MCP converte PDF in immagini
        # 2. Tool interno analizza le immagini

        mock_llm = AsyncMock()
        mock_llm.bind = Mock(return_value=mock_llm)

        # Simula risposta dell'LLM per analisi immagini
        mock_response = Mock()
        mock_response.content = "Il documento contiene un grafico e del testo in italiano."
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        tool = create_multimodal_llm_tool(mock_llm)

        # Simula le immagini restituite da un tool MCP
        mcp_output_images = [
            base64.b64encode(b"page_1_content").decode('utf-8'),
            base64.b64encode(b"page_2_content").decode('utf-8')
        ]

        # L'agent invoca il tool interno
        result = asyncio.run(tool.ainvoke({
            "prompt": "Analizza queste pagine del PDF e descrivi il contenuto",
            "images_base64": mcp_output_images,
            "max_tokens": 2048
        }))

        # Verifica il risultato
        self.assertIn("grafico", result)
        self.assertIn("italiano", result)

        # Verifica che l'LLM sia stato chiamato con entrambe le immagini
        self.mock_llm.ainvoke.assert_called_once()


def run_tests():
    """
    Funzione helper per eseguire i test
    """
    # VECCHIO CODICE (commentato):
    # unittest.main()

    # NUOVO CODICE: Configurazione test runner con verbosità
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    # Esegui i test
    print("\n" + "=" * 80)
    print("UNIT TESTS - Tool Multimodale Interno")
    print("=" * 80 + "\n")

    success = run_tests()

    print("\n" + "=" * 80)
    if success:
        print("✅ TUTTI I TEST SONO PASSATI")
    else:
        print("❌ ALCUNI TEST SONO FALLITI")
    print("=" * 80 + "\n")

    exit(0 if success else 1)
