import {ChatWrapper} from "../ChatWrapper.js";
import {
    ChatModelFunctions, ChatWrapperGenerateContextStateOptions, ChatWrapperGeneratedContextState, ChatWrapperSettings
} from "../types.js";
import {SpecialToken, LlamaText, SpecialTokensText} from "../utils/LlamaText.js";
import {ChatModelFunctionsDocumentationGenerator} from "./utils/ChatModelFunctionsDocumentationGenerator.js";

// source: https://ai.google.dev/gemma/docs/formatting
// source: https://www.promptingguide.ai/models/gemma
export class GemmaChatWrapper extends ChatWrapper {
    public readonly wrapperName: string = "Gemma";

    public override readonly settings: ChatWrapperSettings = {
        supportsSystemMessages: false,
        functions: {
            call: {
                optionalPrefixSpace: true,
                prefix: '{"name": "',
                paramsPrefix: '", "parameters": ',
                suffix: "}",
                emptyCallParamsPlaceholder: {}
            },
            result: {
                prefix: LlamaText(new SpecialTokensText("\n"), "Result: "),
                suffix: LlamaText(new SpecialTokensText("\n"))
            }
        }
    };

    public override generateAvailableFunctionsSystemText(availableFunctions: ChatModelFunctions, {documentParams = true}: {
        documentParams?: boolean
    }) {
        const functionsDocumentationGenerator = new ChatModelFunctionsDocumentationGenerator(availableFunctions);

        if (!functionsDocumentationGenerator.hasAnyFunctions)
            return LlamaText([]);

        return LlamaText.joinValues("\n", [
            "You have access to the following functions. To call a function, respond with JSON for a function call.",
            'Respond in the format {"name": function name, "parameters": function call parameters}.',
            "Do not use variables.",
            "",
            functionsDocumentationGenerator.getLlama3_2LightweightFunctionSignatures({documentParams}),
            "",
            "After calling a function, the result will appear afterwards and is only visible to you.",
            "To make information visible to the user, you must include it in your response.",
            "Only call functions when needed."
        ]);
    }

    public override generateContextState({
        chatHistory, availableFunctions, documentFunctionParams
    }: ChatWrapperGenerateContextStateOptions): ChatWrapperGeneratedContextState {
        const historyWithFunctions = this.addAvailableFunctionsSystemMessageToHistory(chatHistory, availableFunctions, {
            documentParams: documentFunctionParams
        });

        const resultItems: Array<{
            user: LlamaText,
            model: LlamaText
        }> = [];

        let systemTexts: LlamaText[] = [];
        let userTexts: LlamaText[] = [];
        let modelTexts: LlamaText[] = [];
        let currentAggregateFocus: "system" | "user" | "model" | null = null;

        function flush() {
            if (systemTexts.length > 0 || userTexts.length > 0 || modelTexts.length > 0) {
                const systemText = LlamaText.joinValues("\n\n", systemTexts);
                let userText = LlamaText.joinValues("\n\n", userTexts);

                // there's no system prompt support in Gemma, so we'll prepend the system text to the user message
                if (systemText.values.length > 0) {
                    if (userText.values.length === 0)
                        userText = systemText;
                    else
                        userText = LlamaText([
                            systemText,
                            "\n\n---\n\n",
                            userText
                        ]);
                }

                resultItems.push({
                    user: userText,
                    model: LlamaText.joinValues("\n\n", modelTexts)
                });
            }

            systemTexts = [];
            userTexts = [];
            modelTexts = [];
        }

        for (const item of historyWithFunctions) {
            if (item.type === "system") {
                if (currentAggregateFocus !== "system")
                    flush();

                currentAggregateFocus = "system";
                systemTexts.push(LlamaText.fromJSON(item.text));
            } else if (item.type === "user") {
                if (currentAggregateFocus !== "system" && currentAggregateFocus !== "user")
                    flush();

                currentAggregateFocus = "user";
                userTexts.push(LlamaText(item.text));
            } else if (item.type === "model") {
                currentAggregateFocus = "model";
                modelTexts.push(this.generateModelResponseText(item.response));
            } else
                void (item satisfies never);
        }

        flush();

        const contextText = LlamaText(
            new SpecialToken("BOS"),
            resultItems.map(({user, model}, index) => {
                const isLastItem = index === resultItems.length - 1;

                return LlamaText([
                    (user.values.length === 0)
                        ? LlamaText([])
                        : LlamaText([
                            new SpecialTokensText("<start_of_turn>user\n"),
                            user,
                            new SpecialTokensText("<end_of_turn>\n")
                        ]),

                    (model.values.length === 0 && !isLastItem)
                        ? LlamaText([])
                        : LlamaText([
                            new SpecialTokensText("<start_of_turn>model\n"),
                            model,

                            isLastItem
                                ? LlamaText([])
                                : new SpecialTokensText("<end_of_turn>\n")
                        ])
                ]);
            })
        );

        return {
            contextText,
            stopGenerationTriggers: [
                LlamaText(new SpecialToken("EOS")),
                LlamaText(new SpecialTokensText("<end_of_turn>\n")),
                LlamaText("<end_of_turn>")
            ]
        };
    }
}
