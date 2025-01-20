import {
  bigIntToHex,
  bytesToHex,
  privateToAddress,
  toBytes,
} from "@nomicfoundation/ethereumjs-util";

import { defaultHardhatNetworkParams } from "hardhat/internal/core/config/default-config";
import {
  MempoolOrder,
  TracingConfig,
} from "hardhat/internal/hardhat-network/provider/node-types";
import { EdrProviderWrapper } from "hardhat/internal/hardhat-network/provider/provider";
import { LoggerConfig } from "hardhat/internal/hardhat-network/provider/modules/logger";
import { SolidityStackTrace } from "hardhat/internal/hardhat-network/stack-traces/solidity-stack-trace";
import { Response } from "@ignored/edr-optimism";

function toBuffer(x: Parameters<typeof toBytes>[0]) {
  return Buffer.from(toBytes(x));
}

const abi = require("ethereumjs-abi");

const senderPrivateKey =
  "0xe331b6d69882b4cb4ea581d88e0b604039a3de5967688d3dcffdd2270c0fd109";

const senderAddress = bytesToHex(privateToAddress(toBuffer(senderPrivateKey)));

export async function instantiateProvider(
  loggerConfig: LoggerConfig,
  tracingConfig: TracingConfig
): Promise<EdrProviderWrapper> {
  const config = {
    hardfork: "cancun",
    chainId: 1,
    networkId: 1,
    blockGasLimit: 10_000_000,
    minGasPrice: 0n,
    automine: true,
    intervalMining: 0,
    mempoolOrder: "priority" as MempoolOrder,
    chains: defaultHardhatNetworkParams.chains,
    genesisAccounts: [
      {
        privateKey: senderPrivateKey,
        balance: 1e15,
      },
    ],
    allowUnlimitedContractSize: false,
    throwOnTransactionFailures: true,
    throwOnCallFailures: false,
    allowBlocksWithSameTimestamp: false,
    coinbase: "0x0000000000000000000000000000000000000000",
    initialBaseFeePerGas: 0,
    enableTransientStorage: false,
    enableRip7212: false,
  };

  const provider = await EdrProviderWrapper.create(
    config,
    loggerConfig,
    tracingConfig
  );

  return provider;
}

export function encodeConstructorParams(
  contractAbi: any[],
  params: any[]
): Buffer {
  const fAbi = contractAbi.find((a) => a.type === "constructor");

  if (fAbi === undefined || params.length === 0) {
    return Buffer.from([]);
  }

  const types = fAbi.inputs.map((i: any) => i.type);

  return abi.rawEncode(types, params);
}

export function encodeCall(
  contractAbi: any[],
  functionName: string,
  params: any[]
): Buffer {
  const fAbi = contractAbi.find(
    (a) => a.name === functionName && a.inputs.length === params.length
  );

  const types = fAbi.inputs.map((i: any) => i.type);
  const methodId = abi.methodID(functionName, types);

  return Buffer.concat([methodId, abi.rawEncode(types, params)]);
}

interface TxData {
  data: Buffer;
  to?: Buffer;
  value?: bigint;
  gas?: bigint;
}

/**
 * Returns a SolidityStackTrace object if the transaction failed, the contract address if
 * the transaction successfully deployed a contract, or undefined if the transaction
 * succeeded and didn't deploy a contract.
 */
export async function traceTransaction(
  provider: EdrProviderWrapper,
  txData: TxData
): Promise<SolidityStackTrace | string | undefined> {
  const stringifiedArgs = JSON.stringify({
    method: "eth_sendTransaction",
    params: [
      {
        from: senderAddress,
        data: bytesToHex(txData.data),
        to: txData.to !== undefined ? bytesToHex(txData.to) : undefined,
        value: bigIntToHex(txData.value ?? 0n),
        // If the test didn't define a gasLimit, we assume 4M is enough
        gas: bigIntToHex(txData.gas ?? 4000000n),
        gasPrice: bigIntToHex(10n),
      },
    ],
  });

  const responseObject: Response =
    // eslint-disable-next-line @typescript-eslint/dot-notation
    await provider["_provider"].handleRequest(stringifiedArgs);

  let response;
  if (typeof responseObject.data === "string") {
    response = JSON.parse(responseObject.data);
  } else {
    response = responseObject.data;
  }

  const receipt: any = await provider.request({
    method: "eth_getTransactionReceipt",
    params: [response.result ?? response.error.data.transactionHash],
  });

  const stackTrace: SolidityStackTrace | null = responseObject.stackTrace();

  const contractAddress = receipt.contractAddress?.slice(2);

  if (stackTrace === null) {
    return contractAddress;
  }

  return stackTrace;
}
