/**
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import crypto from 'crypto';
import fs from 'fs';
import { appendFile, readFile, rm, writeFile } from 'fs/promises';
import os from 'os';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import { CreateDatasetRequest, UpdateDatasetRequest } from '../types';
import {
  Dataset,
  DatasetMetadata,
  DatasetMetadataSchema,
  DatasetStore,
  EvalFlowInputSchema,
} from '../types/eval';
import { logger } from '../utils/logger';

/**
 * A local, file-based DatasetStore implementation.
 */
export class LocalFileDatasetStore implements DatasetStore {
  private readonly storeRoot;
  private readonly indexFile;
  private readonly INDEX_DELIMITER = '\n';
  private static cachedDatasetStore: LocalFileDatasetStore | null = null;

  private constructor() {
    this.storeRoot = this.generateRootPath();
    this.indexFile = this.getIndexFilePath();
    fs.mkdirSync(this.storeRoot, { recursive: true });
    if (!fs.existsSync(this.indexFile)) {
      fs.writeFileSync(path.resolve(this.indexFile), '');
    }
    logger.info(
      `Initialized local file dataset store at root: ${this.storeRoot}`
    );
  }

  static getDatasetStore() {
    if (!this.cachedDatasetStore) {
      this.cachedDatasetStore = new LocalFileDatasetStore();
    }
    return this.cachedDatasetStore;
  }

  static reset() {
    this.cachedDatasetStore = null;
  }

  async createDataset(req: CreateDatasetRequest): Promise<DatasetMetadata> {
    return this.createDatasetInternal(req.data, req.displayName);
  }

  private async createDatasetInternal(
    data: Dataset,
    displayName?: string
  ): Promise<DatasetMetadata> {
    const datasetId = this.generateDatasetId();
    const filePath = path.resolve(
      this.storeRoot,
      this.generateFileName(datasetId)
    );

    if (!fs.existsSync(filePath)) {
      logger.error(`Dataset already exists at ` + filePath);
      throw new Error(
        `Create dataset failed: file already exists at {$filePath}`
      );
    }

    logger.info(`Saving Dataset to ` + filePath);
    await writeFile(filePath, JSON.stringify(data));

    const metadata = {
      datasetId: datasetId,
      size: Array.isArray(data) ? data.length : data.samples.length,
      uri: 'file://' + filePath,
      version: 1,
      displayName: displayName,
      createTime: Date.now().toString(),
      updateTime: Date.now().toString(),
    };

    logger.debug(
      `Saving DatasetMetadata for ID ${datasetId} to ` +
        path.resolve(this.indexFile)
    );
    await appendFile(
      path.resolve(this.indexFile),
      JSON.stringify(metadata) + this.INDEX_DELIMITER
    );
    return metadata;
  }

  async updateDataset(req: UpdateDatasetRequest): Promise<DatasetMetadata> {
    if (!req.datasetId) {
      return this.createDatasetInternal(req.patch, req.displayName);
    }
    const datasetId = req.datasetId;
    const filePath = path.resolve(
      this.storeRoot,
      this.generateFileName(datasetId)
    );
    if (!fs.existsSync(filePath)) {
      return this.createDatasetInternal(req.patch, req.displayName);
    }

    const metadatas = await this.listDatasets();
    const prevMetadata = metadatas.find(
      (metadata) => metadata.datasetId === datasetId
    );
    if (!prevMetadata) {
      throw new Error(`Update dataset failed: dataset metadata not found`);
    }

    logger.info(`Updating Dataset at ` + filePath);
    await writeFile(filePath, JSON.stringify(req.patch));

    const newMetadata = {
      datasetId: datasetId,
      size: Array.isArray(req.patch)
        ? req.patch.length
        : req.patch.samples.length,
      uri: 'file://' + filePath,
      version: prevMetadata.version++,
      displayName: req.displayName,
      createTime: prevMetadata.createTime,
      updateTime: Date.now().toString(),
    };

    logger.debug(
      `Updating DatasetMetadata for ID ${datasetId} at ` +
        path.resolve(this.indexFile)
    );
    const prevMetadataIndex = metadatas.indexOf(prevMetadata);
    // Replace the metadata object with the new metadata
    metadatas[prevMetadataIndex] = newMetadata;
    await writeFile(
      path.resolve(this.indexFile),
      metadatas
        .map((metadata) => JSON.stringify(metadata))
        .join(this.INDEX_DELIMITER)
    );

    return newMetadata;
  }

  async getDataset(id: string): Promise<Dataset> {
    const filePath = path.resolve(this.storeRoot, this.generateFileName(id));
    if (!fs.existsSync(filePath)) {
      throw new Error(`Dataset not found for dataset ID {$id}`);
    }
    return await readFile(filePath, 'utf8').then((data) =>
      EvalFlowInputSchema.parse(JSON.parse(data))
    );
  }

  async listDatasets(): Promise<DatasetMetadata[]> {
    let metadatas = await readFile(this.indexFile, 'utf8').then((data) => {
      if (!data) {
        return [];
      }
      // strip the final carriage return before parsing all lines
      return data
        .slice(0, -1)
        .split(this.INDEX_DELIMITER)
        .map(this.parseLineToDatasetMetadata);
    });

    logger.debug(`Found Dataset Metadatas: ${JSON.stringify(metadatas)}`);

    return metadatas;
  }

  async deleteDataset(id: string): Promise<void> {
    const filePath = path.resolve(this.storeRoot, this.generateFileName(id));
    if (!fs.existsSync(filePath)) {
      return;
    }
    return await rm(filePath);
  }

  private generateRootPath(): string {
    const rootHash = crypto
      .createHash('md5')
      .update(process.cwd() || 'unknown')
      .digest('hex');
    return path.resolve(os.tmpdir(), `.genkit/${rootHash}/datasets`);
  }

  private generateDatasetId(): string {
    return uuidv4();
  }

  private generateFileName(datasetId: string): string {
    return `${datasetId}.json`;
  }

  private getIndexFilePath(): string {
    return path.resolve(this.storeRoot, 'index.txt');
  }

  private parseLineToDatasetMetadata(key: string): DatasetMetadata {
    return DatasetMetadataSchema.parse(JSON.parse(key));
  }
}
