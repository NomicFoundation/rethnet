import { JsonStreamStringify } from "json-stream-stringify";

import {
  ContractAndFunctionName,
  GENERIC_CHAIN_TYPE,
  genericChainProviderFactory,
  MineOrdering,
  SubscriptionEvent,
} from "..";
import { ALCHEMY_URL, getContext, isCI } from "./helpers";

describe("Provider", () => {
  const context = getContext();

  before(async () => {
    await context.registerProviderFactory(
      GENERIC_CHAIN_TYPE,
      genericChainProviderFactory()
    );
  });

  const providerConfig = {
    allowBlocksWithSameTimestamp: false,
    allowUnlimitedContractSize: true,
    bailOnCallFailure: false,
    bailOnTransactionFailure: false,
    blockGasLimit: 300_000_000n,
    chainId: 1n,
    chains: [],
    coinbase: Buffer.from("0000000000000000000000000000000000000000", "hex"),
    enableRip7212: false,
    hardfork: "Latest",
    initialBlobGas: {
      gasUsed: 0n,
      excessGas: 0n,
    },
    initialParentBeaconBlockRoot: Buffer.from(
      "0000000000000000000000000000000000000000000000000000000000000000",
      "hex"
    ),
    minGasPrice: 0n,
    mining: {
      autoMine: true,
      memPool: {
        order: MineOrdering.Priority,
      },
    },
    networkId: 123n,
    ownedAccounts: [
      {
        secretKey:
          "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
        balance: 1000n * 10n ** 18n,
      },
    ],
  };

  const loggerConfig = {
    enable: false,
    decodeConsoleLogInputsCallback: (_inputs: Buffer[]): string[] => {
      return [];
    },
    getContractAndFunctionNameCallback: (
      _code: Buffer,
      _calldata?: Buffer
    ): ContractAndFunctionName => {
      return {
        contractName: "",
      };
    },
    printLineCallback: (_message: string, _replace: boolean) => {},
  };

  it("issue 543", async function () {
    if (ALCHEMY_URL === undefined || !isCI()) {
      this.skip();
    }

    // This test is slow because the debug_traceTransaction is performed on a large transaction.
    this.timeout(1_800_000);

    const provider = await context.createProvider(
      GENERIC_CHAIN_TYPE,
      {
        fork: {
          jsonRpcUrl: ALCHEMY_URL,
        },
        genesisState: [],
        initialBaseFeePerGas: 0n,
        ...providerConfig,
      },
      loggerConfig,
      {
        subscriptionCallback: (_event: SubscriptionEvent) => {},
      },
      {}
    );

    const debugTraceTransaction = `{
        "jsonrpc": "2.0",
        "method": "debug_traceTransaction",
        "params": ["0x7e460f200343e5ab6653a8857cc5ef798e3f5bea6a517b156f90c77ef311a57c"],
        "id": 1
      }`;

    const response = await provider.handleRequest(debugTraceTransaction);

    let responseData;

    if (typeof response.data === "string") {
      responseData = JSON.parse(response.data);
    } else {
      responseData = response.data;
    }

    // Validate that we can query the response data without crashing.
    const _json = new JsonStreamStringify(responseData);
  });

  it("issue 771", async function () {
    const provider = await context.createProvider(
      GENERIC_CHAIN_TYPE,
      {
        genesisState: [],
        initialBaseFeePerGas: 0n,
        ...providerConfig,
        mining: {
          // Enable interval mining to validate that provider shutdown works correctly
          interval: 1n,
          ...providerConfig.mining,
        },
      },
      loggerConfig,
      {
        subscriptionCallback: (_event: SubscriptionEvent) => {},
      },
      {}
    );

    // Make a dummy request to ensure the provider constructor doesn't become a no-op
    await provider.handleRequest(
      JSON.stringify({
        id: 1,
        jsonrpc: "2.0",
        method: "eth_blockNumber",
        params: [],
      })
    );
  });
});
