document.addEventListener('alpine:init', () => {
    Alpine.data('chatInterface', () => ({
        messages: [],
        userInput: '',
        isTyping: false,
        isMultimodal: false,
        processingSteps: [],
        currentStepText: '',

        init() {
            this.fetchHistory();
            this.$watch('messages', () => {
                this.$nextTick(() => {
                    const el = document.getElementById('chat-messages');
                    if (el) {
                        el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
                    }
                    if (window.lucide) window.lucide.createIcons();
                });
            });
        },

        async fetchHistory() {
            try {
                const res = await fetch('/api/v1/chat/history');
                const data = await res.json();
                this.messages = data.reverse().map(chat => ([
                    { id: chat.id + '-q', type: 'user', content: chat.question },
                    { id: chat.id + '-a', type: 'bot', content: chat.answer, sources: chat.sources, response_time: chat.response_time_ms }
                ])).flat();
            } catch (err) {
                console.error("Failed to load history", err);
            }
        },

        async sendMessage() {
            if (!this.userInput.trim() || this.isTyping) return;
            
            const q = this.userInput;
            this.userInput = '';
            this.messages.push({ id: Date.now(), type: 'user', content: q });
            
            this.isTyping = true;
            this.processingSteps = [
                { name: 'Embedding Query', status: 'active' },
                { name: 'Vector Search', status: 'waiting' },
                { name: 'Pinecone Retrieval', status: 'waiting' },
                { name: 'Gemini Logic', status: 'waiting' }
            ];
            
            this.currentStepText = 'Generating high-dimensional embeddings...';

            try {
                // Simulate step progress for visual effect
                setTimeout(() => { 
                    this.processingSteps[0].status = 'done';
                    this.processingSteps[1].status = 'active';
                    this.currentStepText = 'Querying medical vector database (Pinecone)...';
                }, 800);

                setTimeout(() => { 
                    this.processingSteps[1].status = 'done';
                    this.processingSteps[2].status = 'active';
                    this.currentStepText = 'Retrieving medical context documents...';
                }, 1600);

                const res = await fetch('/api/v1/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: q, is_multimodal: this.isMultimodal })
                });
                
                if (!res.ok) throw new Error("API Error");
                
                const data = await res.json();
                
                this.processingSteps[2].status = 'done';
                this.processingSteps[3].status = 'active';
                this.currentStepText = 'Gemini 3 Flash is synthesizing response...';

                setTimeout(() => {
                    this.messages.push({ 
                        id: data.id, 
                        type: 'bot', 
                        content: data.answer, 
                        sources: data.sources,
                        response_time: data.response_time_ms
                    });
                    this.isTyping = false;
                }, 500);

            } catch (err) {
                this.messages.push({ 
                    id: Date.now() + 1, 
                    type: 'bot', 
                    content: "I'm sorry, I encountered an error processing your request. Please try again later."
                });
                this.isTyping = false;
            }
        },

        renderMarkdown(text) {
            if (window.marked) return window.marked.parse(text);
            return text;
        },

        copyToClipboard(text) {
            navigator.clipboard.writeText(text);
            // Optionally dispatch a toast event
        }
    }));
});
