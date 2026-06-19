export interface Provider {
    id: string;
    resolveUpstreamModel(clientModel?: string): string;
    handleRequest(body: any, headers: Record<string, string>): Promise<Response>;
}
